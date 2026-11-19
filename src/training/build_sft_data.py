"""Turn successful teacher episodes into SFT chat examples, one per valid step.

Each step is replayed through the real shopper with a mock HTTP transport that records the request, so the
prompt is exactly what the model sees at inference (including what Pydantic AI adds).

The target is the logged reply, {"reasoning": ..., "action": ...}. 
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

from src.agents.prompts import SINGLE_AGENT_COT_PROMPT
from src.agents.single_agent_shopper import SingleAgentShopper
from src.environment.webshop_runner import WebshopRunner

PROMPT_VERSION = hashlib.sha256(SINGLE_AGENT_COT_PROMPT.encode()).hexdigest()[:8]
SFT_DATA_DIR = Path(__file__).parents[2] / "data" / "sft" / PROMPT_VERSION

VALIDATION_HOLDOUT_PCT = 0.1
STUB_REPLY = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"reasoning": "", "action": "stub"}'}}],
              "id": "stub", "object": "chat.completion", "created": 0, "model": "stub"}

requests: list[dict] = []

def record_request(request: httpx2.Request) -> httpx2.Response:
    requests.append(json.loads(request.content))
    return httpx2.Response(200, json=STUB_REPLY)


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


if __name__ == "__main__":
    logs = sorted({path for pattern in sys.argv[1:] for path in glob.glob(pattern)})
    assert logs and all(f"_{PROMPT_VERSION}_" in Path(p).name for p in logs), f"need logs made with the current prompt ({PROMPT_VERSION})"
    
    records = [json.loads(line) for path in logs for line in open(path, encoding="utf-8")]
    assert len({r["goal_idx"] for r in records}) == len(records), "the same goal appears in more than one episode"
    
    successes = [r for r in records if r["reward"] == 1.0]

    all_goals = sorted(r["goal_idx"] for r in successes)
    random.Random(0).shuffle(all_goals)
    val_goals = set(all_goals[:round(len(all_goals) * VALIDATION_HOLDOUT_PCT)])

    shopper = SingleAgentShopper()
    stub = OllamaModel("qwen3.5:9b", provider=OllamaProvider(base_url="http://stub/v1", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record_request))),
                       settings=shopper.model.settings, profile=shopper.model.profile)  # same request shape as the real model
    
    SFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with shopper.agent.override(model=stub), open(SFT_DATA_DIR / "train.jsonl", "w") as train, open(SFT_DATA_DIR / "val.jsonl", "w") as val:
        for record in successes:
            for example in episode_examples(shopper, record):
                (val if record["goal_idx"] in val_goals else train).write(json.dumps(example) + "\n")

    print(f"{len(records)} episodes, {len(successes)} successful ({len(val_goals)} held out for validation) -> {SFT_DATA_DIR}")
