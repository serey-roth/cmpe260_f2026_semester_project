"""Compare the baseline and reward-hacking WebShop agents on the same goal.

Run from the project root:

    python -m src.evaluation.compare_ollama_agents

Environment variables:
    OLLAMA_HOST       Ollama URL (default: http://127.0.0.1:11434)
    OLLAMA_MODEL      model tag (default: llama3:8b)
    OLLAMA_GOAL_INDEX WebShop goal index (default: 0)
    OLLAMA_MAX_STEPS  per-episode action limit (default: 15)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from src.agents.reward_hacking import RewardHackingAgent, audit_episode
from src.environment.main import WebshopAgent
from webshop_minimal import WebAgentTextEnv, init_basedir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "comparative results"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")
GOAL_INDEX = int(os.environ.get("OLLAMA_GOAL_INDEX", "0"))
MAX_STEPS = int(os.environ.get("OLLAMA_MAX_STEPS", "15"))


def check_ollama() -> dict[str, Any]:
    """Check that Ollama is reachable and the selected model is installed."""
    try:
        with urlopen(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except (OSError, URLError) as exc:
        return {
            "service_available": False,
            "model_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    names = {
        model.get("name")
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    return {
        "service_available": True,
        "model_available": MODEL in names,
        "available_models": sorted(name for name in names if name),
    }


def run_comparison() -> dict[str, Any]:
    """Run both policies against the same deterministic WebShop goal."""
    readiness = check_ollama()
    result: dict[str, Any] = {
        "test": "baseline_vs_reward_hacking",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ollama_host": OLLAMA_HOST,
        "model": MODEL,
        "goal_index": GOAL_INDEX,
        "max_steps": MAX_STEPS,
        "readiness": readiness,
    }
    if not readiness["service_available"] or not readiness["model_available"]:
        result["status"] = "blocked"
        result["reason"] = (
            "Start Ollama and run `ollama pull "
            f"{MODEL}` before running this comparison."
        )
        return result

    init_basedir("small")
    started = time.perf_counter()
    baseline_env = WebAgentTextEnv(
        observation_mode="text_rich",
        human_goals=True,
    )
    red_team_env = WebAgentTextEnv(
        observation_mode="text_rich",
        human_goals=True,
        server=baseline_env.server,
    )
    baseline_agent = WebshopAgent(model=MODEL)
    baseline_agent.env = baseline_env
    baseline = baseline_agent.run_episode(
        goal_idx=GOAL_INDEX, max_steps=MAX_STEPS
    )
    baseline["agent"] = "baseline"
    baseline["audit"] = audit_episode(
        baseline_env,
        float(baseline["reward"]),
        baseline.get("reward_info"),
        baseline.get("purchase"),
    )
    red_team = RewardHackingAgent(model=MODEL, env=red_team_env).run_episode(
        goal_idx=GOAL_INDEX, max_steps=MAX_STEPS
    )
    result.update(
        {
            "status": "completed",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "baseline": baseline,
            "reward_hacking": red_team,
            "comparison": summarize_comparison(baseline, red_team),
        }
    )
    return result


def summarize_comparison(
    baseline: dict[str, Any],
    red_team: dict[str, Any],
) -> dict[str, Any]:
    """Return compact metrics that expose proxy/reality divergence."""
    baseline_audit = baseline["audit"]
    red_audit = red_team["audit"]
    return {
        "reward_delta_red_team_minus_baseline": (
            red_team["reward"] - baseline["reward"]
        ),
        "fidelity_delta_red_team_minus_baseline": (
            red_audit["fidelity_score"] - baseline_audit["fidelity_score"]
        ),
        "baseline": {
            "reward": baseline["reward"],
            "fidelity_score": baseline_audit["fidelity_score"],
            "classification": baseline_audit["classification"],
            "steps": baseline["num_steps"],
        },
        "reward_hacking": {
            "reward": red_team["reward"],
            "fidelity_score": red_audit["fidelity_score"],
            "classification": red_audit["classification"],
            "steps": red_team["num_steps"],
        },
    }


def write_result(result: dict[str, Any]) -> Path:
    """Write a timestamped comparison result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = RESULTS_DIR / f"agent-comparison-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    comparison = run_comparison()
    path = write_result(comparison)
    print(json.dumps(comparison.get("comparison", comparison), indent=2))
    print(f"Saved comparison: {path}")
