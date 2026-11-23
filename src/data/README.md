# Data

## What is a goal?

A goal is one shopping task: a human-written instruction plus the product it was written for, with a price cap added by the environment, e.g. *"i need a long clip-in hair extension which is natural looking, and price lower than 30.00 dollars"*. The environment builds 12,087 goals from the human instructions. `env.reset(session=i)` starts goal `i`.

## What are the splits?

The default project configuration uses the bundled `webshop-minimal` small
dataset. It contains 13 human-written goals and already includes search
indexes, so no download or index build is needed for a smoke test. Run
`init_splits` once if you want to create `src/data/splits.jsonl` for the
evaluation runner.

WebShop's official split, by position in the environment's goal list (shuffled with a fixed seed):

| Split | Positions | Goals |
| --- | --- | --- |
| test | 0 to 499 | up to 13 |
| dev | 500 to 1499 | 0 for the small dataset |
| train | 1500 to the end | 0 for the small dataset |

A task (same product, same instruction, same price cap) that appears in more than one split is kept only in the first of test, dev, train, so test is never reduced.

## Full-catalog data (optional)

The Google Drive links in WebShop's setup script are dead so we download the full catalog from Hugging Face.

```bash
uv run hf download YWZBrandon/webshop-data \
    items_shuffle.json items_ins_v2.json items_human_ins.json \
    --repo-type dataset --local-dir src/data/webshop-data
```

## Generate the splits

```bash
uv run python -c "from src.data.splits import init_splits; init_splits()"
```

## Get the splits

```python
from src.data.splits import get_train_splits, get_dev_splits, get_test_splits, load_splits

rows = get_dev_splits()             # same as load_splits(split="dev")
rows[0]                             # {"idx": 500, "asin": "B07YZ3JVL2", "instruction": "..."}
env.reset(session=rows[0]["idx"])   # idx is the goal's position in the environment's goal list
```

## Build the search index

The indexes shipped with WebShop only contain 1,000 products, but for the entire catalog, we need to build our own:

```bash
uv run python -m src.data.search_index                      # 100,000 products: every product a goal was written for, plus a random sample
uv run python -m src.data.search_index --num-docs 2000000   # the whole catalog
```

It writes the index to `src/data/search-index/index` which is used to build the env for the full catalog.

## Provisional SFT demonstrations

The project does not yet specify a shared SFT format. The following local
pipeline produces a provisional format that must be reconciled with the
group's eventual training schema before use. It streams the full catalog into
a 100,000-product search subset, retaining all human-goal products and the
original prices, goal ordering, and split assignments. Search results are
from this subset, not the entire catalog.

With the full-catalog files above available and an Ollama model installed:

```bash
uv run python -m src.training.prepare_catalog
uv run python -m src.training.generate_sft index
uv run python -m src.training.generate_sft generate --model gemma3:4b --count 100
```

Set `PYTHONIOENCODING=utf-8` if the terminal cannot print product titles.
Preparation and generation refuse to overwrite existing output directories;
use `--output-dir` to choose another location. Prepared data and output are
ignored by Git under `src/data/sft-prepared/`.

The generator attempts the first 100 deduplicated **train** goals by default;
it never selects dev or test goals. `run/attempts.jsonl` contains the complete
episode records, including failed attempts. `run/sft.jsonl` contains one
training decision per line, with `messages` (user prompt and assistant action),
`goal_idx`, `step`, and `reward`. `run/config.json` records the model, split,
seed, selection, and acceptance rule.

Only completed trajectories that purchase the target product with reward 1,
a price strictly below any stated cap, valid actions, and no repeated action
are exported. Add `--inspect-products` to prompt the model to inspect each
product and require it to inspect the purchased product. This changes the
prompts saved in the examples. These are automatic filters, not proof of
demonstration quality: inspect examples for correct product attributes,
options, and decision-making before training. The number of accepted
trajectories may be much smaller than the number of attempted goals.
