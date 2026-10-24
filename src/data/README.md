# Data

## What is a goal?

A goal is one shopping task: a human-written instruction plus the product it was written for, with a price cap added by the environment, e.g. *"i need a long clip-in hair extension which is natural looking, and price lower than 30.00 dollars"*. The environment builds 12,087 goals from the human instructions. `env.reset(session=i)` starts goal `i`.

## What are the splits?

WebShop's official split, by position in the environment's goal list (shuffled with a fixed seed):

| Split | Positions | Goals |
| --- | --- | --- |
| test | 0 to 499 | 500 |
| dev | 500 to 1499 | 998 |
| train | 1500 to the end | 10,538 |

A task (same product, same instruction, same price cap) that appears in more than one split is kept only in the first of test, dev, train, so test is never reduced.

## Download the catalog

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
