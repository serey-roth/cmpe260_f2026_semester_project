import argparse
import json
from datetime import datetime
from pathlib import Path

from src.agents.main import EPISODE_LOG_DIR, REPLY_FORMATS, WebshopAgent
from src.data.search_index import INDEX_DIR
from src.data.splits import DATA_DIR, SEED, SPLITS_PATH, init_splits, load_splits
from src.environment.env import build_env

EVAL_RUN_DIR = Path(__file__).parent / "runs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"], help="which split to evaluate on")
    parser.add_argument("--n", type=int, default=None, help="number of goals to run (default: all)")
    parser.add_argument("--model", default="llama3:8b", help="ollama model name")
    parser.add_argument("--prompt", default="direct", choices=list(REPLY_FORMATS), help="direct action or chain of thought")
    parser.add_argument("--resume", type=Path, default=None, help="episode log of an interrupted run; skips the goals it already has")
    args = parser.parse_args()

    if not SPLITS_PATH.exists():
        init_splits()
    rows = load_splits(split=args.split)[:args.n]
    env = build_env(data_dir=DATA_DIR, seed=SEED)
    agent = WebshopAgent(env=env, model=args.model, prompt_style=args.prompt)

    timestamp = datetime.now()
    run_name = f"{args.split}_{args.model.replace(':', '-').replace('/', '-')}_{args.prompt}_{timestamp:%Y%m%d_%H%M%S}"

    EPISODE_LOG_DIR.mkdir(exist_ok=True)
    log_path = EPISODE_LOG_DIR / f"{run_name}.jsonl"

    records = []
    if args.resume:
        log_path, run_name = args.resume, args.resume.stem
        records = [json.loads(line) for line in open(log_path, encoding="utf-8")]
        done_goals = {r["goal_idx"] for r in records}
        rows = [row for row in rows if row["idx"] not in done_goals]

    mode = "a" if args.resume else "w"
    with open(log_path, mode, encoding="utf-8") as log, open(log_path.with_suffix(".md"), mode, encoding="utf-8") as transcript:
        for row in rows:
            record = agent.run_episode(goal_idx=row["idx"])
            shown = record["instruction"].removeprefix("Instruction:").strip()  # the env's page has an "Instruction:" heading
            assert shown == row["instruction"], f"goal {row['idx']} is not the goal the split was made for"
            log.write(json.dumps(record) + "\n")
            log.flush()  # keep the finished episodes if the run is interrupted
            records.append(record)

    EVAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_path = EVAL_RUN_DIR / f"{run_name}.jsonl"

    n = len(records)
    option_scores = [r["reward_info"]["r_option"] for r in records if r["done"] and "r_option" in (r["reward_info"] or {})]
    run = {
        "type": "baseline_single_agent",
        "model": agent.model,
        "prompt": agent.prompt_style,
        "split": args.split,
        "num_goals": n,
        "timestamp": f"{timestamp:%Y%m%d_%H%M%S}",
        "average_reward": sum(r["reward"] for r in records) / n,
        "average_steps": sum(r["num_steps"] for r in records) / n,
        "success_rate": sum(r["reward"] == 1.0 for r in records) / n,  # bought exactly the right product
        "purchase_rate": sum(bool(r["done"]) for r in records) / n,  # bought anything
        "option_score": sum(option_scores) / len(option_scores) if option_scores else None,  # over purchases that had options to pick
    }

    with open(run_path, "w", encoding="utf-8") as run_file:
        run_file.write(json.dumps(run))

    print(run)
