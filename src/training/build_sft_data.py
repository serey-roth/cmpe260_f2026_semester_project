"""Turn successful teacher episodes into SFT chat examples.

Each step is replayed through the real shopper with a mock HTTP transport that records the request, so the
prompt is exactly what the model sees at inference (including what Pydantic AI adds).

- JSON shopper (default): one example per valid step; the target is the logged reply, {"reasoning": ..., "action": ...}.
- Tool shopper (--tools): one example per episode, the whole tool-calling conversation; the stub replies with each
  logged step as a tool call, and the pages come back as tool results.
"""
import src.pyserini_offline  # noqa: F401  before anything imports pyserini

import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import httpx2
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.agents.prompts import SINGLE_AGENT_COT_PROMPT, SINGLE_AGENT_TOOL_PROMPT
from src.agents.single_agent_shopper import AgentModel, SingleAgentShopper
from src.agents.tool_shopper import ToolShopper
from src.environment.webshop_runner import WebshopRunner

PROMPT_VERSION = hashlib.sha256(SINGLE_AGENT_COT_PROMPT.encode()).hexdigest()[:8]  # the teacher's logs
TOOLS = "--tools" in sys.argv
SFT_DATA_DIR = Path(__file__).parents[2] / "data" / "sft" / hashlib.sha256((SINGLE_AGENT_TOOL_PROMPT if TOOLS else SINGLE_AGENT_COT_PROMPT).encode()).hexdigest()[:8]

VALIDATION_HOLDOUT_PCT = 0.1
STUB_REPLY = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"reasoning": "", "action": "stub"}'}}],
              "id": "stub", "object": "chat.completion", "created": 0, "model": "stub"}

requests: list[dict] = []
stub_reply = {"reply": STUB_REPLY}  # the tool converter swaps in each logged step as a tool call

def record_request(request: httpx2.Request) -> httpx2.Response:
    requests.append(json.loads(request.content))
    return httpx2.Response(200, json=stub_reply["reply"])


def episode_examples(shopper: SingleAgentShopper, record: dict) -> list[dict]:
    examples, history = [], []
    for step in record["steps"]:
        if step["is_valid"]:  # ignored steps are not trained on, but stay in the history as the model saw them
            requests.clear()
            shopper.act(record["instruction"], step["observation"], step["available_actions"], list(history))
            target = json.dumps({"reasoning": step["thought"], "action": step["action"]})
            examples.append({"messages": requests[0]["messages"] + [{"role": "assistant", "content": target}]})
        history.append(WebshopRunner.history_entry(step["action"], step["is_valid"]))
    return examples


def episode_conversation(shopper: ToolShopper, record: dict) -> dict | None:
    """The whole episode as one conversation, or None if a step has no tool call to write (no valid reply was logged).
    Steps the page did not allow stay in, as the model saw them: their tool result says nothing happened."""
    requests.clear()
    shopper.reset()
    history: list[str] = []
    for n, step in enumerate(record["steps"]):
        match = WebshopRunner.ACTION_PATTERN.fullmatch(step["action"])
        if match is None:
            return None
        tool, argument = match.group(1).lower(), match.group(2)
        call = {"id": f"call_{n}", "type": "function", "function": {"name": tool, "arguments": json.dumps(
            {"reasoning": step["thought"], "keywords" if tool == "search" else "value": argument})}}
        stub_reply["reply"] = {**STUB_REPLY, "choices": [{"index": 0, "finish_reason": "tool_calls",
                                                    "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
        shopper.act(record["instruction"], step["observation"], step["available_actions"], list(history))
        history.append(WebshopRunner.history_entry(step["action"], step["is_valid"]))

    messages = requests[-1]["messages"] + [stub_reply["reply"]["choices"][0]["message"]]
    for message in messages:  # Qwen's chat template reads tool-call arguments as a mapping, not a JSON string
        for tool_call in message.get("tool_calls") or []:
            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
    return {"messages": messages, "tools": requests[-1]["tools"]}


if __name__ == "__main__":
    logs = sorted({path for pattern in sys.argv[1:] if pattern != "--tools" for path in glob.glob(pattern)})
    assert logs and all(f"_{PROMPT_VERSION}_" in Path(p).name for p in logs), f"need logs made with the current prompt ({PROMPT_VERSION})"
    
    records = [json.loads(line) for path in logs for line in open(path, encoding="utf-8")]
    assert len({r["goal_idx"] for r in records}) == len(records), "the same goal appears in more than one episode"
    
    successes = [r for r in records if r["reward"] == 1.0]

    all_goals = sorted(r["goal_idx"] for r in successes)
    random.Random(0).shuffle(all_goals)
    val_goals = set(all_goals[:round(len(all_goals) * VALIDATION_HOLDOUT_PCT)])

    shopper = (ToolShopper if TOOLS else SingleAgentShopper)(AgentModel("qwen3.5:9b", thinking=True))  # requests go to the stub
    stub = OllamaModel("qwen3.5:9b", provider=OllamaProvider(base_url="http://stub/v1", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record_request))),
                       settings=shopper.model.settings, profile=shopper.model.profile)  # same request shape as the real model
    
    SFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with shopper.agent.override(model=stub), open(SFT_DATA_DIR / "train.jsonl", "w") as train, open(SFT_DATA_DIR / "val.jsonl", "w") as val:
        for record in successes:
            examples = [episode_conversation(shopper, record)] if TOOLS else episode_examples(shopper, record)
            for example in filter(None, examples):
                (val if record["goal_idx"] in val_goals else train).write(json.dumps(example) + "\n")

    print(f"{len(records)} episodes, {len(successes)} successful ({len(val_goals)} held out for validation) -> {SFT_DATA_DIR}")
