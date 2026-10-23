from src.evaluation.compare_ollama_agents import summarize_comparison
import pytest


def test_summarize_comparison_exposes_reward_and_fidelity_delta():
    baseline = {
        "reward": 0.6,
        "num_steps": 5,
        "audit": {
            "fidelity_score": 1.0,
            "classification": "aligned",
        },
    }
    red_team = {
        "reward": 0.8,
        "num_steps": 7,
        "audit": {
            "fidelity_score": 0.33,
            "classification": "reward_hack",
        },
    }

    summary = summarize_comparison(baseline, red_team)

    assert summary["reward_delta_red_team_minus_baseline"] == pytest.approx(0.2)
    assert summary["fidelity_delta_red_team_minus_baseline"] == pytest.approx(-0.67)
    assert summary["reward_hacking"]["classification"] == "reward_hack"
