"""Run the baseline WebshopAgent over a data split and save every episode.

Usage (from repo root, after the data setup in src/data/README.md):
    uv run python -m src.evaluation.run_baseline --split dev --n 50
    uv run python -m src.evaluation.run_baseline --split test            # all 500 test goals

Use dev while developing; keep test for final numbers.
Episodes are appended to a JSONL file one at a time, so a crash or Ctrl+C
keeps everything finished so far. Re-running with the same --out file skips
goals that are already in it (resume).
"""
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from src.agents.main import WebshopAgent
from src.data.search_index import INDEX_DIR
from src.data.splits import DATA_DIR, SEED, SPLITS, load_splits
from src.environment.env import build_env

RESULTS_DIR = Path("src/evaluation/results")


def load_done_goals(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["goal_idx"] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--n", type=int, default=None, help="run only the first N goals of the split")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--out", type=Path, default=None, help="JSONL output path (reuse to resume)")
    args = parser.parse_args()

    rows = load_splits(split=args.split)
    if args.n is not None:
        rows = rows[: args.n]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / (
        f"baseline_{args.split}_n{len(rows)}_s{args.max_steps}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )

    env = build_env(data_dir=DATA_DIR, seed=SEED, index_dir=INDEX_DIR / "index")
    agent = WebshopAgent(env=env)

    done = load_done_goals(out)
    todo = [r for r in rows if r["idx"] not in done]
    print(f"Split={args.split} | writing to {out} | {len(done)} already done, {len(todo)} to run")

    with open(out, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            goal_idx = row["idx"]
            print(f"\n[{i}/{len(todo)}] goal {goal_idx}")
            t0 = time.time()
            try:
                record = agent.run_episode(goal_idx=goal_idx, max_steps=args.max_steps)
                record["error"] = None
            except Exception as e:  # keep going; log the failure instead
                traceback.print_exc()
                record = {"goal_idx": goal_idx, "instruction": row.get("instruction"),
                          "reward": 0.0, "done": False, "num_steps": 0, "error": repr(e)}
            record["split"] = args.split
            record["max_steps"] = args.max_steps
            record["duration_s"] = round(time.time() - t0, 2)
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    print(f"\nSaved episodes to {out}")
    print(f"Summarize with: uv run python -m src.evaluation.summarize {out}")


if __name__ == "__main__":
    main()
