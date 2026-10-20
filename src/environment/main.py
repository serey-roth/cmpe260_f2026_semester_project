import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.webshop.webshop_minimal import WebAgentTextEnv, init_basedir
from ollama import Client

ACTION_PATTERN = re.compile(r"(search|click)\[(.+?)\]", re.IGNORECASE | re.DOTALL)

PROMPT_TEMPLATE = """
You are a personal shopper browsing an ecommerce website. Find and buy the product that best matches the user's request.

The best match MUST satisfy the following conditions:
1. It matches what the user asked for.
2. It has every feature the user asked for (check the title, and open Description or Features if the title does not confirm it).
3. Every option the user asked for (such as color or size) is selected on the product page.
4. Its price is within any price limit in the request.

How to proceed:
- On a search results page, compare titles and prices, then click the ASIN of the best match. Use next > only if nothing fits.
- On a product page, select the requested options. Once the product fits the request and its options are selected, click[buy now].
- "< prev" and "back to search" leave the current page. Only use them if this product does not fit the request.
- Do not reopen a product you already looked at unless you are choosing between products.

User request: 
{user_request}
            
Current page:
{observations}

Valid actions on this page:
- {search_hint}
- click[<value>] where <value> is exactly one of: {clickables}

Your recent actions: 
{recent_history}
You have {remaining_steps} actions left, and the task only counts if you finish by clicking buy now.

Reply with exactly one action and nothing else, and ALWAYS include the square brackets.
Examples: search[blue wireless headphones], click[b09q55jlcs], or click[buy now]
"""

MAX_STEPS = 15

class WebshopAgent:
    def __init__(self):
        self.client = Client()
        self.model = "llama3:8b"
        
        self.env = WebAgentTextEnv(observation_mode="text_rich", human_goals=True)
        
    def run_episode(
        self,
        goal_idx,
        max_steps=MAX_STEPS
    ) -> dict[str, Any]:
        self.env.reset(session=goal_idx)
        instruction = self.env.instruction_text
        obs = self.env.observation 
        
        print(f"{instruction}")
    
        history = []
        steps = []
        reward, done, info = 0.0, False, None
        
        for step in range(1, max_steps + 1):
            available_actions = self.env.get_available_actions()
            action, reply, prompt = self.act(instruction, obs, available_actions, history)
            details = self._get_action_details( action)
            
            print(self.format_step(step, action, details))
    
            next_obs, reward, done, info = self.env.step(action)
            steps.append({
                "step": step,
                "observation": obs,
                "reply": reply,
                "action": action,
                "details": details,
                "reward": reward,
                "done": done,
            })
            
            history.append(action)
            obs = next_obs
            if done:
                break
    
        record = {
            "goal_idx": goal_idx,
            "instruction": instruction,
            "model": agent.model,
            "prompt": prompt,
            "reward": reward,
            "done": done,
            "num_steps": len(steps),
            "reward_info": info,
            "steps": steps,
        }
        if done:
            purchase = self._get_purchase()
            print(purchase)
            record.update({ "purchase": purchase })
            print(f"  bought: {purchase['title']} (${purchase['price']}) options={purchase['options']}")
        
        print(f"  done={done} reward={reward}")
        return record

    def act(
        self,
        user_request,
        observations,
        available_actions,
        history=(),
    ) -> tuple[str, str, str]:
        clickables = available_actions["clickables"]
        search_hint = (
            "search[<keywords>] is allowed." if available_actions["has_search_bar"] else "search is not allowed on this page."
        )
        recent = "\n".join(history[-5:]) or "(none)"
        steps_left = MAX_STEPS - len(history)

        prompt = PROMPT_TEMPLATE.format(
            user_request=user_request,
            observations=observations,
            search_hint=search_hint,
            recent_history=recent,
            remaining_steps=steps_left,
            clickables=clickables
        )

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0, "num_ctx": 8192},
        )
        reply = response["response"]
        
        action = self.parse_action(reply, available_actions)
        return action, reply, prompt # TODO: obfuscate prompt

    @staticmethod
    def parse_action(text, available_actions) -> str:
        match = ACTION_PATTERN.search(text)
        if match is not None:
            kind, arg = match.group(1).lower(), match.group(2).strip()
            return f"{kind}[{arg.lower() if kind == 'click' else arg}]"

        stripped = text.strip().strip("`'\"").strip()
        if available_actions["has_search_bar"] and stripped.lower().startswith("search"):
            query = stripped[len("search"):].strip(" :")
            if query:
                return f"search[{query}]"
            
        lowered = stripped.lower()
        for clickable in sorted(available_actions["clickables"], key=len, reverse=True):
            if clickable != "search" and clickable in lowered:
                return f"click[{clickable}]"
            
        return "click[]"
    
    @staticmethod
    def format_step(step, action, details) -> str:
        kind = details["type"]
        if kind == "product":
            return f"  step {step}: {action} -> {details['title']} (${details['price']})"
        if kind == "option":
            return f"  step {step}: {action} -> option {details['option']} = {details['value']}"
        return f"  step {step}: {action}"

    def _get_action_details(self, action) -> dict[str, Any]:
        match = ACTION_PATTERN.fullmatch(action)
        if match is None:
            return {"type": "invalid"}
        
        kind, arg = match.group(1).lower(), match.group(2)
        if kind == "search":
            return {"type": "search", "query": arg}

        product = self.env.server.product_item_dict.get(arg.upper())
        if product is not None:
            return {
                "type": "product",
                "asin": arg.upper(),
                "title": product["Title"],
                "price": self.env.server.product_prices.get(arg.upper()),
            }
            
        tag = (self.env.text_to_clickable or {}).get(arg)
        if tag is not None and tag.name == "input":
            return {"type": "option", "option": tag.get("name"), "value": arg}
        return {"type": "nav", "target": arg}

    def _get_purchase(self) -> dict[str, Any]:
        session = self.env.server.user_sessions[self.env.session]
        asin = session["asin"]
        
        return {
            "asin": asin,
            "title": self.env.server.product_item_dict[asin]["Title"],
            "price": self.env.server.product_prices.get(asin),
            "options": dict(session["options"]),
        }

EPISODE_LOG_DIR = Path("src/environment/logs")

if __name__ == "__main__":
    init_basedir("small")
    agent = WebshopAgent()

    EPISODE_LOG_DIR.mkdir(exist_ok=True)
    
    goal_idx = 0
    episode_name = f"episodes_{datetime.now():%Y%m%d_%H%M%S}_goal_{goal_idx}"
    stem = EPISODE_LOG_DIR / episode_name
    
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as jsonl:
        record = agent.run_episode(goal_idx=0)
        jsonl.write(json.dumps(record) + "\n")
        
    print(f"Saved episode {stem}.jsonl")
