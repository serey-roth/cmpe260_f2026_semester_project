import argparse
import gc
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from src.data.configuration import configure_data
from webshop_minimal.engine import load_products
from webshop_minimal.utils import get_file_path, get_human_attr_path

from src.data.splits import DATA_DIR, SEED

INDEX_DIR = Path(__file__).parent / "search-index"
NUM_DOCS = 100000


def get_product_details(product: dict) -> dict:
    """Return a product's search document: its `id` and the `contents` text that is searched. Use the same text as
    webshop/setup_search_engine.py, but leave out the full product."""
    option_texts = []
    for option_name, option_contents in product.get("options", {}).items():
        option_texts.append(f"{option_name}: {', '.join(option_contents)}")
    option_text = ", and ".join(option_texts)

    contents = " ".join([product["Title"], product["Description"], product["BulletPoints"][0], option_text])
    return {"id": product["asin"], "contents": contents.lower()}


def select_asins(all_products: list[dict], target_asins: set[str], num_docs: int, seed: int) -> set[str]:
    """Select every target product, plus sampled products to make up for `num_docs` in total."""
    targets = {p["asin"] for p in all_products if p["asin"] in target_asins}
    others = [p["asin"] for p in all_products if p["asin"] not in target_asins]
    return targets | set(random.Random(seed).sample(others, max(0, num_docs - len(targets))))


def build_index(num_docs: int = NUM_DOCS, index_dir: Path = INDEX_DIR, threads: int = 1) -> None:
    """Load the catalog, write the selected products as search documents, and index them with pyserini."""
    configure_data(DATA_DIR, SEED)
    start = time.time()
    all_products, *_ = load_products(filepath=get_file_path(), human_goals=True)
    print(f"loaded {len(all_products):,} products ({time.time() - start:.0f}s)", flush=True)

    target_asins = set(json.load(open(get_human_attr_path())))
    if num_docs >= len(all_products):
        selected = {p["asin"] for p in all_products}
    else:
        selected = select_asins(all_products, target_asins, num_docs, SEED)

    docs_dir = index_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    with open(docs_dir / "documents.jsonl", "w", encoding="utf-8") as f:
        for product in all_products:  # catalog order
            if product["asin"] in selected:
                f.write(json.dumps(get_product_details(product)) + "\n")
    print(f"wrote {len(selected):,} documents ({time.time() - start:.0f}s)", flush=True)

    # delete the entire product catalog from memory for indexing
    del all_products
    gc.collect()

    # same command and flags as webshop/scripts/run_indexing.sh
    subprocess.run([
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(docs_dir),
        "--index", str(index_dir / "index"),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ], check=True)
    print(f"indexed {len(selected):,} documents ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=NUM_DOCS, help="documents to index; at least the number of target products")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    build_index(args.num_docs, args.index_dir, args.threads)
