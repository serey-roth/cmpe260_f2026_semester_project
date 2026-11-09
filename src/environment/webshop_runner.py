import re
import textwrap
from typing import Any

from src.agents.agent_shopper_protocol import MAX_STEPS, AgentShopperAction, AgentShopperProtocol
from webshop_minimal import WebAgentTextEnv
from webshop_minimal.engine import parse_action


class WebshopRunner:
    ACTION_PATTERN = re.compile(r"^(search|click)\[(.*)\]$", re.IGNORECASE)

    def __init__(
        self, 
        env: WebAgentTextEnv, 
        shopper: AgentShopperProtocol,
    ):
        self.env = env
        self.shopper = shopper

    def run_episode(
        self,
        goal_idx,
        max_steps=MAX_STEPS
    ) -> dict[str, Any]:
        self.env.reset(session=goal_idx)
        self.shopper.reset()
        instruction = self.env.instruction_text
        obs = self.env.observation 
        
        print(f"{instruction}")
    
        history: list[str] = []
        steps = []
        reward, done, info = 0.0, False, None
        
        for step in range(1, max_steps + 1):
            available_actions = self.env.get_available_actions()
            selection = self.act(instruction, obs, available_actions, history)
            details = self._get_action_details(selection.action)
            is_valid_selection = self.is_valid_action(selection.action, available_actions)

            print(self.format_step(step, selection.action, details))
            if selection.reasoning:
                print(textwrap.indent(textwrap.fill(selection.reasoning, width=110), "      | "))

            next_obs, reward, done, info = self.env.step(selection.action)
            steps.append({
                "step": step,
                "observation": obs,
                "available_actions": available_actions,  # with the observation and history, rebuilds the step's prompt for SFT
                "thought": selection.reasoning,
                "action": selection.action,
                "is_valid": is_valid_selection,
                "details": details,
                "reward": reward,
                "done": done,
            })
            
            history.append(
                selection.action if is_valid_selection 
                else f"{selection.action} (Nothing happened. Use search[<keywords>] or click[<value>] with a listed value)"
            )
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
        user_request: str,
        observations: str,
        available_actions: dict[str, Any],
        history: list[str] = [],
    ) -> AgentShopperAction:
        return self.shopper.act(
            user_request=user_request,
            current_page=observations,
            available_actions=available_actions,
            history=list(history),
        )
    
    @staticmethod
    def format_step(step, action, details) -> str:
        kind = details["type"]
        if kind == "product":
            return f"  step {step}: {action} -> {details['title']} (${details['price']})"
        if kind == "option":
            return f"  step {step}: {action} -> option {details['option']} = {details['value']}"
        return f"  step {step}: {action}"

    def _get_action_details(self, action) -> dict[str, Any]:
        match = self.ACTION_PATTERN.fullmatch(action)
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
        
    def is_valid_action(self, action: str, available_actions: dict[str, Any]) -> bool:
        name, arg = parse_action(action)
        if arg is None:
            return False
        if name == "search":
            return True  # the env runs a search from any page, with or without a search bar
        return name == "click" and arg.lower() != "search" and arg.lower() in available_actions["clickables"]
