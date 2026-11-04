import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.search_index import INDEX_DIR
from src.data.splits import DATA_DIR, SEED
from src.environment.env import build_env
from webshop_minimal import WebAgentTextEnv
import httpx
from ollama import Client

ACTION_PATTERN = re.compile(r"(search|click)\[(.+?)\]", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<(think|reasoning)>(.*?)(?:</\1>|$)", re.IGNORECASE | re.DOTALL)
THINK_END_PATTERN = re.compile(r"</(?:think|reasoning)>", re.IGNORECASE)

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
- Options reset when you leave a product page. Only options shown as [clicked button] on the current page are selected.
- "< prev" and "back to search" leave the current page. Only use them if this product does not fit the request.
- Do not reopen a product you already looked at UNLESS you are choosing between products.
- Buy the product as soon as the product fits and the desired options are selected. DO NOT SEARCH AGAIN FOR SOMETHING BETTER.

User request: 
{user_request}
            
Current page:
{observations}

Valid actions on this page:
- {search_hint}
- click[<value>] where <value> is exactly one of: {clickables}

Your actions so far: 
{recent_history}
You have {remaining_steps} actions left, and the task only counts if you finish by clicking buy now.

{reply_format}
"""

DIRECT_REPLY_FORMAT = """Reply with exactly one action and nothing else, and ALWAYS include the square brackets.
Examples: search[blue wireless headphones], click[b09q55jlcs], or click[buy now]"""

COT_REPLY_FORMAT = """First reason briefly inside <reasoning></reasoning> tags, then write exactly one action on the line after </reasoning>.
In your reasoning, compare the user request with the current page: Is this the right product? Which requested options are already selected and which are still missing? Is the price within the limit?
The action must be written as: search[<keywords>] or click[<value>].

Example reply on a product page:
<reasoning>
This is a black relaxed-fit sweatpant under $40, which matches. The request asks for black and xx-large. Neither is selected yet, so I select black first.
</reasoning>
click[black]

Example reply on the search page:
<reasoning>
I have not searched yet, so I search for the key words of the request.
</reasoning>
search[xx-large black relaxed fit workout sweatpants]"""

REPLY_FORMATS = {"direct": DIRECT_REPLY_FORMAT, "cot": COT_REPLY_FORMAT}

MAX_STEPS = 25
MAX_RETRIES = 5

class WebshopAgent:
    def __init__(self, env: WebAgentTextEnv, model: str = "llama3:8b", prompt_style: str = "direct"):
        if prompt_style not in REPLY_FORMATS:
            raise ValueError(f"prompt_style must be one of {list(REPLY_FORMATS)}")
        self.client = Client(timeout=120)
        self.model = model
        self.prompt_style = prompt_style
        self.env = env
        
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
            action, reply = self.act(instruction, obs, available_actions, history)
            details = self._get_action_details( action)
            thought = self.extract_thought(reply)

            print(self.format_step(step, action, details))
            if thought:
                print(textwrap.indent(textwrap.fill(thought, width=110), "      | "))

            next_obs, reward, done, info = self.env.step(action)
            steps.append({
                "step": step,
                "observation": obs,
                "reply": reply,
                "thought": thought,
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
    ) -> tuple[str, str]:
        clickables = available_actions["clickables"]
        search_hint = (
            "search[<keywords>] is allowed." if available_actions["has_search_bar"] else "search is not allowed on this page."
        )
        recent = "\n".join(history) or "(none)"
        steps_left = MAX_STEPS - len(history)

        prompt = PROMPT_TEMPLATE.format(
            user_request=user_request,
            observations=observations,
            search_hint=search_hint,
            recent_history=recent,
            remaining_steps=steps_left,
            clickables=clickables,
            reply_format=REPLY_FORMATS[self.prompt_style],
        )
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": 0, "num_ctx": 8192},
                    # turn model default thinking off (too slow)
                    # use cot prompt instead
                    think=False,  
                )
                
            except httpx.TimeoutException:
                print(f"  agent timed out (attempt {attempt}/{MAX_RETRIES})")
                continue

            reply = response["response"]
            if response.get("thinking"):  # keep it in the log if a model thinks anyway
                reply = f"<think>\n{response['thinking']}\n</think>\n{reply}"
            if reply.strip():
                break
            
            print(f"  empty reply from {self.model} (attempt {attempt}/{MAX_RETRIES})")
        else:
            raise RuntimeError(f"no reply from {self.model} after {MAX_RETRIES} attempts")

        # only parse the action after the reasoning tags
        action_text = THINK_END_PATTERN.split(reply)[-1]
        action = self.parse_action(action_text, available_actions)
        return action, reply

    @staticmethod
    def extract_thought(reply) -> str | None:
        match = THINK_PATTERN.search(reply)
        return match.group(2).strip() if match else None

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

EPISODE_LOG_DIR = Path(__file__).parent / "logs"

if __name__ == "__main__":
    env = build_env(
        data_dir=DATA_DIR,
        seed=SEED,
        index_dir=INDEX_DIR / "index"
    )
    agent = WebshopAgent(env=env)

    EPISODE_LOG_DIR.mkdir(exist_ok=True)
    
    goal_idx = 0
    episode_name = f"episodes_{datetime.now():%Y%m%d_%H%M%S}_goal_{goal_idx}"
    stem = EPISODE_LOG_DIR / episode_name
    
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as jsonl:
        record = agent.run_episode(goal_idx=0)
        jsonl.write(json.dumps(record) + "\n")
        
    print(f"Saved episode {stem}.jsonl")
