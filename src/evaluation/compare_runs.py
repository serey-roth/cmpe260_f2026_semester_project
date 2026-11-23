"""Combine per-slice dev summaries into one row per model and print a comparison table.

Usage (from repo root, on the HPC or your Mac):
    python3 compare_runs.py src/evaluation/runs/dev_single_*
Plain Python 3, no packages needed. Slices of the same model/thinking setting are merged,
weighted by goals. Only the latest run of each slice start is used if a slice was rerun.
"""
import json
import sys
from collections import defaultdict

FIELDS = ["average_reward", "success_rate", "purchase_rate", "option_score", "average_steps",
          "invalid_actions_per_episode", "no_option_purchases"]
BREAKDOWN = ["r_type", "r_att", "r_option", "r_price"]

runs = {}  # (model, thinking, slice start) -> latest summary
for path in sys.argv[1:]:
    name = path.rsplit("/", 1)[-1]
    start = name.rsplit("_from", 1)[-1].split(".")[0]
    try:
        with open(path) as f:
            r = json.loads(f.read().strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        continue
    key = (r["model"], r["thinking"], start)
    if key not in runs or r["timestamp"] > runs[key]["timestamp"]:
        runs[key] = r

groups = defaultdict(list)
for (model, thinking, _), r in runs.items():
    groups[(model, thinking)].append(r)

def merged(rs, getter):
    n = sum(r["num_goals"] for r in rs)
    return sum(getter(r) * r["num_goals"] for r in rs) / n

rows = []
for (model, thinking), rs in groups.items():
    n = sum(r["num_goals"] for r in rs)
    row = {"model": model, "thinking": "on" if thinking else "off", "goals": n}
    for k in FIELDS:
        row[k] = merged(rs, lambda r, k=k: r[k]) if k != "no_option_purchases" else sum(r[k] for r in rs)
    for k in BREAKDOWN:
        row[k] = merged(rs, lambda r, k=k: r["reward_breakdown"][k])
    rows.append(row)
rows.sort(key=lambda r: (r["model"], r["thinking"]))

cols = [("model", "Model", 24), ("thinking", "Think", 5), ("goals", "Goals", 5), ("average_reward", "Reward", 6),
        ("success_rate", "Success", 7), ("purchase_rate", "Bought", 6), ("average_steps", "Steps", 5),
        ("r_option", "r_opt", 5), ("r_att", "r_att", 5), ("r_type", "r_typ", 5), ("invalid_actions_per_episode", "Invalid", 7)]
print(" ".join(f"{h:<{w}}" for _, h, w in cols))
for r in rows:
    print(" ".join(f"{r[k]:<{w}.3f}" if isinstance(r[k], float) else f"{str(r[k]):<{w}}" for k, _, w in cols))
