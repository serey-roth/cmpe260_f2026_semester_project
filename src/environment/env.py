from pathlib import Path

import webshop_minimal.env as env_module
from pyserini.search.lucene import LuceneSearcher
from webshop_minimal import WebAgentTextEnv

from src.data.configuration import configure_data


def build_env(data_dir: Path, seed: int, index_dir: Path | None = None, **env_kwargs) -> WebAgentTextEnv:
    """Build the WebShop env on the catalog in `data_dir`, searching `index_dir` if given."""
    configure_data(data_dir, seed)
    if index_dir is not None:
        env_module.init_search_engine = lambda num_products=None: LuceneSearcher(str(index_dir))

    return WebAgentTextEnv(**{"observation_mode": "text_rich", "human_goals": True, **env_kwargs})
