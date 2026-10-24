import argparse
import json
from datetime import datetime

from src.agents.main import EPISODE_LOG_DIR, WebshopAgent
from src.data.search_index import INDEX_DIR
from src.data.splits import DATA_DIR, SEED, get_test_splits
from src.environment.env import build_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="number of test goals to run (default: all)")
    args = parser.parse_args()

    rows = get_test_splits()[:args.n]
    env = build_env(data_dir=DATA_DIR, seed=SEED, index_dir=INDEX_DIR / "index")
    agent = WebshopAgent(env=env)

    EPISODE_LOG_DIR.mkdir(exist_ok=True)
    log_path = EPISODE_LOG_DIR / f"dev_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    records = []
    with open(log_path, "w", encoding="utf-8") as log:
        for row in rows:
            record = agent.run_episode(goal_idx=row["idx"])
            shown = record["instruction"].removeprefix("Instruction:").strip()  # the env's page has an "Instruction:" heading
            assert shown == row["instruction"], f"goal {row['idx']} is not the goal the split was made for"
            log.write(json.dumps(record) + "\n")
            log.flush()  # keep the finished episodes if the run is interrupted
            records.append(record)

    n = len(records)
    print(f"\n{agent.model} on {n} dev goals")
    print(f"average reward: {sum(r['reward'] for r in records) / n:.3f}")
    print(f"average steps:  {sum(r['num_steps'] for r in records) / n:.2f}")
    print(f"bought something in {sum(r['done'] for r in records)} of {n} episodes")
    print(f"episodes saved to {log_path}")
