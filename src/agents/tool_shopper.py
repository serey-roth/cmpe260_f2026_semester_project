from typing import Any

from pydantic_ai import Agent as PydanticAgent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.exceptions import ModelTimeoutError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from src.agents.agent_shopper_protocol import MAX_RETRIES, MAX_STEPS, AgentShopperAction, AgentShopperProtocol
from src.agents.prompts import SINGLE_AGENT_TOOL_PROMPT
from src.agents.single_agent_shopper import AgentModel

REASONING = {"type": "string", "description": "Why this action, at most 3 short sentences."}
TOOLS = [  # reasoning first, as in the JSON reply, so the model reasons before it picks the action
    ToolDefinition(name="search", description="Search the shop for products. Works on a page with a search bar.", parameters_json_schema={
        "type": "object", "required": ["reasoning", "keywords"],
        "properties": {"reasoning": REASONING, "keywords": {"type": "string", "description": "What to search for."}}}),
    ToolDefinition(name="click", description="Click a product ID, option, button or link on the current page.", parameters_json_schema={
        "type": "object", "required": ["reasoning", "value"],
        "properties": {"reasoning": REASONING, "value": {"type": "string", "description": "Exactly one of the click values listed for the current page."}}}),
]
INVALID_NOTE = "Nothing happened: that action is not available on this page. Use search or click with a listed value.\n"


def page_text(current_page: str, available_actions: dict[str, Any], steps_left: int) -> str:
    clickables = [value for value in available_actions["clickables"] if value != "search"]  # the search button: use the search tool
    return (
        f"Current page text: {current_page}\n"
        f"Click values: {', '.join(clickables)}. "
        f"{'Use the search tool to search.' if available_actions['has_search_bar'] else 'search is not allowed on this page.'}\n"
        f"You have {steps_left} actions left, and the task only counts if you finish by clicking buy now."
    )


class ToolShopper(AgentShopperProtocol):
    def __init__(self, model: AgentModel):
        self.model = model
        self.agent = PydanticAgent(model=self.model, instructions=SINGLE_AGENT_TOOL_PROMPT, output_type=[DeferredToolRequests, str],
                                   toolsets=[ExternalToolset(TOOLS)], retries=MAX_RETRIES)
        self.reset()

    def reset(self) -> None:
        self.messages: list[ModelMessage] = []
        self.pending: list[str] = []  # ids of the last reply's tool calls, answered by the next act()

    def act(
        self,
        user_request: str,
        current_page: str,
        available_actions: dict[str, Any],
        history: list[str] = [],
    ) -> AgentShopperAction:
        page = page_text(current_page, available_actions, MAX_STEPS - len(history))
        user_prompt, tool_results = None, None
        if self.pending:  # the page after the last action is that tool call's result; extra calls in one reply are ignored
            note = INVALID_NOTE if history and "Nothing happened" in history[-1] else ""
            ignored = {call_id: "Ignored: call one tool per step." for call_id in self.pending[1:]}
            tool_results = DeferredToolResults(calls={self.pending[0]: note + page, **ignored})
        else:  # first step, or after a reply without a tool call
            user_prompt = page if self.messages else f"User request: {user_request}\n{page}"

        try:
            result = self.agent.run_sync(user_prompt, message_history=self.messages, deferred_tool_results=tool_results)
        except (ModelTimeoutError, UnexpectedModelBehavior) as e:
            print(f"  no valid reply: {e}")
            return AgentShopperAction(action="invalid", reasoning="The agent failed to produce a valid action.")

        self.messages = result.all_messages()
        if isinstance(result.output, str):  # replied without a tool call: nothing to execute
            self.pending = []
            return AgentShopperAction(action="invalid", reasoning=result.output)
        calls = result.output.calls
        self.pending = [call.tool_call_id for call in calls]
        args = calls[0].args_as_dict()
        argument = args.get("keywords") if calls[0].tool_name == "search" else args.get("value")
        if argument is None:
            print(f"  unreadable tool call: {calls[0].tool_name} {calls[0].args}")
        action = f"{calls[0].tool_name}[{argument}]" if argument is not None else "invalid"
        return AgentShopperAction(action=action, reasoning=args.get("reasoning", ""))
