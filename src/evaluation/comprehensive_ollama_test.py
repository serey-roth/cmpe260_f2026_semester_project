"""Run and log one complete Ollama-backed WebShop evaluation episode.

Run from the project root after starting Ollama and pulling the configured
model:

    python src/evaluation/comprehensive_ollama_test.py

The test uses goal 0 and the agent's normal 15-step limit by default. Set
OLLAMA_MODEL or OLLAMA_GOAL_INDEX to change those values.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.main import WebshopAgent


RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")
GOAL_INDEX = int(os.environ.get("OLLAMA_GOAL_INDEX", "0"))
MAX_STEPS = int(os.environ.get("OLLAMA_MAX_STEPS", "15"))


def check_ollama() -> dict[str, Any]:
    """Return service and model readiness without making an inference call."""
    try:
        with urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except (OSError, URLError) as exc:
        return {
            "service_available": False,
            "model_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    model_names = {
        model.get("name")
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    return {
        "service_available": True,
        "model_available": MODEL in model_names,
        "available_models": sorted(name for name in model_names if name),
    }


def run() -> int:
    """Run one episode, write JSON and text logs, and return a shell status."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    readiness = check_ollama()
    result: dict[str, Any] = {
        "test": "comprehensive_ollama_episode",
        "started_at": started_at.isoformat(),
        "ollama_host": OLLAMA_HOST,
        "model": MODEL,
        "goal_index": GOAL_INDEX,
        "max_steps": MAX_STEPS,
        "readiness": readiness,
    }

    if not readiness["service_available"] or not readiness["model_available"]:
        result["status"] = "blocked"
        result["reason"] = (
            "Start Ollama and pull the configured model before running this test."
        )
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = 0.0
        _write_results(result)
        print(json.dumps(result, indent=2))
        return 2

    try:
        started = time.perf_counter()
        agent = WebshopAgent()
        agent.model = MODEL
        record = agent.run_episode(goal_idx=GOAL_INDEX, max_steps=MAX_STEPS)
        duration = time.perf_counter() - started
        result.update(
            {
                "status": "passed" if record["done"] else "completed_without_purchase",
                "duration_seconds": round(duration, 3),
                "episode": record,
            }
        )
        return_code = 0 if record["done"] else 1
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return_code = 1

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result.setdefault("duration_seconds", 0.0)
    _write_results(result)
    print(json.dumps(result, indent=2))
    return return_code


def _write_results(result: dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"ollama-test-{timestamp}.json"
    text_path = RESULTS_DIR / f"ollama-test-{timestamp}.txt"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(_format_summary(result), encoding="utf-8")
    print(f"JSON result: {json_path}")
    print(f"Text result: {text_path}")


def _format_summary(result: dict[str, Any]) -> str:
    lines = [
        "Comprehensive Ollama Evaluation",
        "================================",
        f"Status: {result['status']}",
        f"Model: {result['model']}",
        f"Goal index: {result['goal_index']}",
        f"Maximum steps: {result['max_steps']}",
        f"Duration seconds: {result['duration_seconds']}",
        "",
        "Readiness:",
        json.dumps(result["readiness"], indent=2),
    ]
    episode = result.get("episode")
    if episode:
        lines.extend(
            [
                "",
                "Episode:",
                f"  Done: {episode['done']}",
                f"  Reward: {episode['reward']}",
                f"  Steps: {episode['num_steps']}",
            ]
        )
    if result.get("reason"):
        lines.extend(["", f"Reason: {result['reason']}"])
    if result.get("error"):
        lines.extend(["", f"Error: {result['error']}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(run())
