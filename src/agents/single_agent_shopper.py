from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent as PydanticAgent, NativeOutput, RunContext
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.ollama import OllamaModel

from src.agents.agent_shopper_protocol import MAX_RETRIES, MAX_STEPS, TIMEOUT_SECONDS, AgentShopperAction, AgentShopperProtocol, run_with_fallback
from src.agents.prompts import SINGLE_AGENT_COT_PROMPT

@dataclass
class ShopperState:
    history: list[str]
    remaining_steps: int


class SingleAgentShopper(AgentShopperProtocol):
    def __init__(
        self, 
        model: Literal["llama3:8b", "qwen3.5:9b"] = "llama3:8b",
    ):
        self.model = OllamaModel(
            model_name=model
        )
        
        self.agent = PydanticAgent(
            model=self.model,
            deps_type=ShopperState,
            instructions=SINGLE_AGENT_COT_PROMPT,
            capabilities=[Thinking(effort=False)],
            output_type=NativeOutput(AgentShopperAction),
            model_settings= {"timeout": TIMEOUT_SECONDS} ,
            retries=MAX_RETRIES
        )
        
        
        @self.agent.instructions
        def get_history(ctx: RunContext[ShopperState]) -> str:
            history = "\n".join(ctx.deps.history) or "(none)"
            return (
                f"Your actions so far:\n{history}\n"
                f"You have {ctx.deps.remaining_steps} actions left, and the task only counts if you finish by clicking buy now."
            )
            
            
    def reset(self) -> None:
        pass

    def act(
        self,
        user_request: str,
        current_page: str,
        available_actions: dict[str, Any],
        history: list[str] = [],
    ) -> AgentShopperAction:
        steps_left = MAX_STEPS - len(history)
        shopper_state = ShopperState(
            history=list(history),
            remaining_steps=steps_left,
        )
        
        user_prompt = (
            f"User request: {user_request}\n"
            f"Current page text: {current_page}\n"
            f"Click values: {', '.join(available_actions['clickables'])}. "
            f"{'search[<keywords>] is allowed.' if available_actions['has_search_bar'] else 'search is not allowed on this page.'}\n"
        )
        
        
        return run_with_fallback(
            self.agent,
            user_prompt,
            fallback=AgentShopperAction(action="invalid", reasoning="The agent failed to produce a valid action."),
            deps=shopper_state,
        )
