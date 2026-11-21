import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import src.pyserini_offline  # noqa: F401  before anything imports pyserini, so it can run without internet
from src.agents.prompts import SINGLE_AGENT_COT_PROMPT
from src.agents.single_agent_shopper import SingleAgentShopper
from src.data.search_index import INDEX_DIR
from src.data.splits import DATA_DIR, SEED, load_splits
from src.environment.env import build_env
from src.environment.webshop_runner import WebshopRunner
from src.evaluation.metrics import summarize

EVAL_RUN_DIR = Path(__file__).parent / "runs"
EPISODE_LOG_DIR = Path(__file__).parents[1] / "agents" / "logs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"], help="which split to evaluate on")
    parser.add_argument("--start", type=int, default=0, help="index of the first goal in the split, to run slices in parallel")
    parser.add_argument("--n", type=int, default=None, help="number of goals to run (default: all)")
    parser.add_argument("--model", default="llama3:8b", help="ollama model name")
    parser.add_argument("--no-thinking", action="store_true", help="turn off the model's native thinking (for the ablation)")
    parser.add_argument("--resume", type=Path, default=None, help="episode log of an interrupted run; skips the goals it already has")
    args = parser.parse_args()

    rows = load_splits(split=args.split)[args.start:][:args.n]
    env = build_env(data_dir=DATA_DIR, seed=SEED, index_dir=INDEX_DIR / "index")
    shopper = SingleAgentShopper(model=args.model, thinking=not args.no_thinking)
    prompt = SINGLE_AGENT_COT_PROMPT
    prompt_version = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    runner = WebshopRunner(env=env, shopper=shopper)

    timestamp = datetime.now()
    run_name = f"{args.split}_single_{args.model.replace(':', '-').replace('/', '-')}_{prompt_version}_{"nothink" if args.no_thinking else "think"}_{timestamp:%Y%m%d_%H%M%S}_from{args.start}"  # start keeps parallel slices in separate files

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

    run = {
        "type": "single_cot",
        "model": args.model,
        "prompt_version": prompt_version,
        "thinking": not args.no_thinking,
        "split": args.split,
        "timestamp": f"{timestamp:%Y%m%d_%H%M%S}",
        **summarize(records),
    }

    with open(run_path, "w", encoding="utf-8") as run_file:
        run_file.write(json.dumps(run))

    print(run)
