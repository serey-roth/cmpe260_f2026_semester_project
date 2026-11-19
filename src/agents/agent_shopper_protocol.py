from dataclasses import dataclass
from typing import Any, Protocol

from typing import Any, TypeVar

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelTimeoutError, UnexpectedModelBehavior

MAX_STEPS = 25
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60
MAX_REPLY_TOKENS = 8192

@dataclass
class AgentShopperAction:
    reasoning: str
    action: str


class AgentShopperProtocol(Protocol):
    def reset(self) -> None: ...

    def act(
        self,
        user_request: str,
        current_page: str,
        available_actions: dict[str, Any],
        history: list[str] = [],
    ) -> AgentShopperAction: ...

OutputT = TypeVar("OutputT")

def run_with_fallback(agent: Agent[Any, OutputT], prompt: str, fallback: OutputT, deps: Any = None) -> OutputT:
    try:
        return agent.run_sync(prompt, deps=deps).output
    except (ModelTimeoutError, UnexpectedModelBehavior) as e:
        print(f"  no valid reply: {e}")
        return fallback
