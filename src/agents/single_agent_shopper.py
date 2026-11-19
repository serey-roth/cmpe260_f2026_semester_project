from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent as PydanticAgent, ModelSettings, NativeOutput, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile

from src.agents.agent_shopper_protocol import MAX_REPLY_TOKENS, MAX_RETRIES, MAX_STEPS, TIMEOUT_SECONDS, AgentShopperAction, AgentShopperProtocol, run_with_fallback
from src.agents.prompts import SINGLE_AGENT_COT_PROMPT

@dataclass
class ShopperState:
    history: list[str]
    remaining_steps: int


class SingleAgentShopper(AgentShopperProtocol):
    def __init__(
        self, 
        model: str = "llama3:8b",
        thinking: bool = True,
    ):
        # Sent in extra_body because Ollama ignores what pydantic-ai sends for these two:
        # - max_tokens: pydantic-ai sends max_completion_tokens. The cap includes thinking,
        #   so a model stuck thinking fails fast instead of filling its context
        # - reasoning_effort "none": pydantic-ai's thinking=False sends nothing, so the model keeps thinking
        extra_body: dict[str, Any] = {"max_tokens": MAX_REPLY_TOKENS}
        if not thinking:
            extra_body["reasoning_effort"] = "none"
        settings = ModelSettings(timeout=TIMEOUT_SECONDS, extra_body=extra_body)

        # Inline system prompts on: otherwise pydantic-ai adds a system message explaining its <system> tags, which we never use
        self.model = OllamaModel(model_name=model, settings=settings, profile=ModelProfile(supports_inline_system_prompts=True))

        self.agent = PydanticAgent(
            model=self.model,
            deps_type=ShopperState,
            output_type=NativeOutput(AgentShopperAction),
            retries=MAX_RETRIES
        )

        @self.agent.instructions
        def system_prompt(ctx: RunContext[ShopperState]) -> str:
            history = "\n".join(ctx.deps.history) or "(none)"
            return (
                f"{SINGLE_AGENT_COT_PROMPT.strip()}\n\n"
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
