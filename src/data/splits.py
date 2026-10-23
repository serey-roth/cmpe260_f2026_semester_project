import hashlib
import json
from pathlib import Path

from src.data.configuration import configure_data
from webshop_minimal import WebAgentTextEnv

DATA_DIR = Path(__file__).parent / "webshop-data"  # see README.md for how to download it
SPLITS_PATH = Path(__file__).parent / "splits.jsonl"
SEED = 233 # env random seed number

SPLITS = ["train", "dev", "test"]
SPLIT_PRIORITY = ["test", "dev", "train"]
SPLIT_RANGES = {
    "test": (0, 500),
    "dev": (500, 1500),
    "train": (1500, None)
}


def goal_task(goal: dict) -> tuple[str, str]:
    return goal["asin"], goal["instruction_text"]


def split_goals(goals: list[dict]) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Split goals into train, dev and test sets with test set taking high priority. Duplicates are removed based on priority."""
    splits, removed, claimed = {}, {}, set()
    for name in SPLIT_PRIORITY:
        lo, hi = SPLIT_RANGES[name]
        idxs = range(lo, len(goals) if hi is None else hi)
        splits[name] = [i for i in idxs if goal_task(goals[i]) not in claimed]
        removed[name] = len(idxs) - len(splits[name])
        claimed |= {goal_task(goals[i]) for i in splits[name]}
    return splits, removed


def save_splits(goals: list[dict], splits: dict[str, list[int]], removed: dict[str, int], path: Path = SPLITS_PATH) -> None:
    meta = {
        "num_goals": len(goals),
        "seed": SEED,
        "ranges": SPLIT_RANGES,
        "sizes": {name: len(idxs) for name, idxs in splits.items()},
        "removed_as_duplicates": removed,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for name, idxs in splits.items():
            for i in idxs:
                asin, instruction = goal_task(goals[i])
                f.write(json.dumps({"split": name, "idx": i, "asin": asin, "instruction": instruction}) + "\n")


def init_splits():
    configure_data(DATA_DIR, SEED)

    with WebAgentTextEnv(observation_mode="text_rich", human_goals=True) as env:
        goals = env.server.goals
        splits, removed = split_goals(goals)
        save_splits(goals, splits, removed)


def load_splits(path: Path = SPLITS_PATH, split: str = "train") -> list[dict]:
    """Load splits, with train splits by default."""
    if not (split in SPLITS):
        raise ValueError(f"The given split is invalid. Split must be one of these: {SPLITS}")

    with open(path, encoding="utf-8") as f:
        f.readline()  # skip the metadata line
        splits = []
        for line in f:
            row = json.loads(line)
            row_split = row.pop("split")
            if row_split != split:
                continue

            splits.append(row)

        return splits


def get_train_splits():
    return load_splits(split="train")


def get_dev_splits():
    return load_splits(split="dev")


def get_test_splits():
    return load_splits(split="test")

