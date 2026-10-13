"""Minimal functional test for the local WebShop text environment."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBSHOP_ROOT = PROJECT_ROOT / "third_party" / "webshop"
DATA_FILE = WEBSHOP_ROOT / "data" / "items_shuffle_1000.json"


def test_webshop_text_environment_can_reset_and_search():
    """Load WebShop, reset it, and execute one valid search action."""
    if not DATA_FILE.exists():
        pytest.fail(
            f"WebShop data is missing: {DATA_FILE}. "
            "Run `conda activate webshop && export PYTHONNOUSERSITE=1 && "
            "bash src/scripts/setup_webshop.sh` from the project root first. "
            "If setup reports a Google Drive download error, the upstream "
            "WebShop data link is unavailable and the test cannot run yet."
        )

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    sys.path.insert(0, str(WEBSHOP_ROOT))

    from web_agent_site.envs import WebAgentTextEnv

    env = WebAgentTextEnv(
        observation_mode="text",
        num_products=1000,
    )
    try:
        observation = env.reset()[0]
        available_actions = env.get_available_actions()

        assert isinstance(observation, str)
        assert observation
        assert available_actions["has_search_bar"] is True

        next_observation, reward, done, info = env.step("search[shoes]")

        assert isinstance(next_observation, str)
        assert next_observation
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert info is None
    finally:
        env.close()
