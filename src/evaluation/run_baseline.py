"""Run the baseline WebshopAgent over many goals and save every episode.

Usage (from repo root):
    uv run python -m src.evaluation.run_baseline --start 0 --n 50 --max-steps 15

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

from webshop_minimal import init_basedir

from src.environment.main import WebshopAgent

RESULTS_DIR = Path("src/evaluation/results")


def load_done_goals(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["goal_idx"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="first goal index")
    parser.add_argument("--n", type=int, default=50, help="number of goals to run")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--out", type=Path, default=None, help="JSONL output path (reuse to resume)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / (
        f"baseline_{datetime.now():%Y%m%d_%H%M%S}_g{args.start}-{args.start + args.n - 1}_s{args.max_steps}.jsonl"
    )

    init_basedir("small")
    agent = WebshopAgent()

    done = load_done_goals(out)
    goals = [g for g in range(args.start, args.start + args.n) if g not in done]
    print(f"Writing to {out} | {len(done)} already done, {len(goals)} to run")

    with open(out, "a", encoding="utf-8") as f:
        for i, goal_idx in enumerate(goals, 1):
            print(f"\n[{i}/{len(goals)}] goal {goal_idx}")
            t0 = time.time()
            try:
                record = agent.run_episode(goal_idx=goal_idx, max_steps=args.max_steps)
                record["error"] = None
            except Exception as e:  # keep going; log the failure instead
                traceback.print_exc()
                record = {"goal_idx": goal_idx, "reward": 0.0, "done": False,
                          "num_steps": 0, "error": repr(e)}
            record["max_steps"] = args.max_steps
            record["duration_s"] = round(time.time() - t0, 2)
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    print(f"\nSaved episodes to {out}")
    print(f"Summarize with: uv run python -m src.evaluation.summarize {out}")


if __name__ == "__main__":
    main()
