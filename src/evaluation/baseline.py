import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.agents.prompts import SINGLE_AGENT_COT_PROMPT
from src.agents.single_agent_shopper import SingleAgentShopper
from src.data.search_index import INDEX_DIR
from src.data.splits import DATA_DIR, SEED, load_splits
from src.environment.env import build_env
from src.environment.webshop_runner import WebshopRunner

EVAL_RUN_DIR = Path(__file__).parent / "runs"
EPISODE_LOG_DIR = Path(__file__).parents[1] / "agents" / "logs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"], help="which split to evaluate on")
    parser.add_argument("--start", type=int, default=0, help="index of the first goal in the split, to run slices in parallel")
    parser.add_argument("--n", type=int, default=None, help="number of goals to run (default: all)")
    parser.add_argument("--model", default="llama3:8b", help="ollama model name")
    parser.add_argument("--resume", type=Path, default=None, help="episode log of an interrupted run; skips the goals it already has")
    args = parser.parse_args()

    rows = load_splits(split=args.split)[args.start:][:args.n]
    env = build_env(data_dir=DATA_DIR, seed=SEED, index_dir=INDEX_DIR / "index")
    shopper = SingleAgentShopper(model=args.model)
    prompt = SINGLE_AGENT_COT_PROMPT
    prompt_version = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    runner = WebshopRunner(env=env, shopper=shopper)

    timestamp = datetime.now()
    run_name = f"{args.split}_single_{args.model.replace(':', '-').replace('/', '-')}_{prompt_version}_{timestamp:%Y%m%d_%H%M%S}_from{args.start}"  # start keeps parallel slices in separate files

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
            record = runner.run_episode(goal_idx=row["idx"])
            shown = record["instruction"].removeprefix("Instruction:").strip()  # the env's page has an "Instruction:" heading
            assert shown == row["instruction"], f"goal {row['idx']} is not the goal the split was made for"
            log.write(json.dumps(record) + "\n")
            log.flush() 
            records.append(record)

    EVAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_path = EVAL_RUN_DIR / f"{run_name}.jsonl"

    def mean(values: list) -> float | None:
        return sum(values) / len(values) if values else None

    n = len(records)
    purchases = [r for r in records if r["done"]]
    option_scores = [r["reward_info"]["r_option"] for r in purchases if "r_option" in (r["reward_info"] or {})]
    viewed = [Counter(s["details"]["asin"] for s in r["steps"] if s["details"]["type"] == "product") for r in records]
    run = {
        "type": "single_cot",
        "model": args.model,
        "prompt_version": prompt_version,
        "split": args.split,
        "num_goals": n,
        "timestamp": f"{timestamp:%Y%m%d_%H%M%S}",
        "average_reward": sum(r["reward"] for r in records) / n,
        "average_steps": sum(r["num_steps"] for r in records) / n,
        "success_rate": sum(r["reward"] == 1.0 for r in records) / n,  # bought exactly the right product
        "purchase_rate": sum(bool(r["done"]) for r in records) / n,  # bought anything
        "option_score": sum(option_scores) / len(option_scores) if option_scores else None,  # over purchases that had options to pick
        "invalid_actions_per_episode": sum(not s.get("is_valid", True) for r in records for s in r["steps"]) / n,  # steps the env ignored
        "loops": sum(max(v.values(), default=0) >= 3 for v in viewed),  # episodes that opened the same product 3+ times
        "no_buy": n - len(purchases),
        "no_option_purchases": sum("r_option" in r["reward_info"] and not r.get("purchase", {}).get("options") for r in purchases),  # needed options, selected none
        "reward_breakdown": {  # means over purchases
            key: mean([float(r["reward_info"][key]) for r in purchases if key in r["reward_info"]])
            for key in ["r_type", "r_att", "r_option", "r_price"]
        },
        "avg_searches": sum(s["details"]["type"] == "search" for r in records for s in r["steps"]) / n,
        "avg_products_viewed": sum(len(v) for v in viewed) / n,  # distinct products opened
    }

    with open(run_path, "w", encoding="utf-8") as run_file:
        run_file.write(json.dumps(run))

    print(run)
