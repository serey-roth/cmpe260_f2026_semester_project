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
from webshop_minimal import init_basedir


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
        init_basedir("small")
        agent = WebshopAgent()
        agent.model = MODEL
        record = agent.run_episode(goal_idx=GOAL_INDEX, max_steps=MAX_STEPS)
        record["verification"] = _build_verification_record(agent, record)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    episode_dir = RESULTS_DIR / f"ollama-test-{timestamp}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    json_path = episode_dir / "episode.json"
    text_path = episode_dir / "summary.txt"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(_format_summary(result), encoding="utf-8")
    print(f"JSON result: {json_path}")
    print(f"Text result: {text_path}")


def _build_verification_record(
    agent: WebshopAgent, record: dict[str, Any]
) -> dict[str, Any]:
    """Capture the goal and purchased product data needed for manual review."""
    session = agent.env.server.user_sessions[agent.env.session]
    goal = session["goal"]
    purchase = record.get("purchase")
    product = (
        agent.env.server.product_item_dict.get(purchase["asin"])
        if purchase is not None
        else None
    )
    reward_info = record.get("reward_info") or {}
    goal_options = goal.get("goal_options") or {}
    requested_options = (
        dict(goal_options) if isinstance(goal_options, dict) else list(goal_options)
    )
    selected_options = dict(session.get("options") or {})
    price = purchase.get("price") if purchase is not None else None
    price_match = reward_info.get("r_price")
    option_match = reward_info.get("r_option")
    verification = {
        "prompt": goal["instruction_text"],
        "requested": {
            "asin": goal.get("asin"),
            "query": goal.get("query"),
            "category": goal.get("category"),
            "product_category": goal.get("product_category"),
            "attributes": list(goal.get("attributes") or []),
            "options": requested_options,
            "price_upper": goal.get("price_upper"),
        },
        "purchase": {
            "asin": purchase.get("asin") if purchase else None,
            "title": product.get("Title") if product else None,
            "price": price,
            "selected_options": selected_options,
            "attributes": list(product.get("Attributes") or []) if product else [],
            "category": product.get("category") if product else None,
            "product_category": product.get("product_category") if product else None,
            "query": product.get("query") if product else None,
            "bullet_points": list(product.get("BulletPoints") or []) if product else [],
            "description": product.get("Description") if product else None,
        },
        "match": {
            "product_type": reward_info.get("r_type") == 1.0,
            "attributes": reward_info.get("r_att") == 1.0,
            "options": option_match is None or option_match == 1.0,
            "price": price_match is True,
            "overall": bool(
                purchase
                and reward_info.get("r_type") == 1.0
                and reward_info.get("r_att") == 1.0
                and (option_match is None or option_match == 1.0)
                and price_match is True
            ),
        },
    }
    return verification


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
        verification = episode.get("verification")
        if verification:
            lines.extend(
                [
                    "",
                    "Verification:",
                    f"  Prompt: {verification['prompt']}",
                    f"  Purchased: {verification['purchase']['title']}",
                    f"  Price: {verification['purchase']['price']}",
                    f"  Selected options: {verification['purchase']['selected_options']}",
                    f"  Attributes match: {verification['match']['attributes']}",
                    f"  Options match: {verification['match']['options']}",
                    f"  Price match: {verification['match']['price']}",
                    f"  Overall match: {verification['match']['overall']}",
                ]
            )
    if result.get("reason"):
        lines.extend(["", f"Reason: {result['reason']}"])
    if result.get("error"):
        lines.extend(["", f"Error: {result['error']}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(run())
