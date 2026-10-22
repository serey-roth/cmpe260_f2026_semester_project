"""Summarize and plot baseline episodes saved by run_baseline.py.

Usage:
    uv run python -m src.evaluation.summarize src/evaluation/results/baseline_*.jsonl
"""
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BREAKDOWN_KEYS = ["r_type", "r_att", "r_option", "r_price"]


def load(paths):
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return records


def summarize(records):
    ok = [r for r in records if not r.get("error")]
    bought = [r for r in ok if r.get("done")]
    rewards = [float(r.get("reward") or 0) for r in ok]
    steps = [r["num_steps"] for r in ok]

    s = {
        "episodes": len(records),
        "errors": len(records) - len(ok),
        "mean_reward": mean(rewards) if rewards else 0,
        "success_rate (reward==1)": float(mean(r == 1.0 for r in rewards)) if rewards else 0,
        "buy_rate (clicked buy now)": len(bought) / len(ok) if ok else 0,
        "mean_steps": mean(steps) if steps else 0,
        "median_steps": median(steps) if steps else 0,
        "mean_steps_when_bought": mean(r["num_steps"] for r in bought) if bought else 0,
        "mean_reward_when_bought": mean(float(r["reward"]) for r in bought) if bought else 0,
        "mean_duration_s": mean(r.get("duration_s", 0) for r in records) if records else 0,
    }
    # Reward breakdown (only present on episodes that bought something)
    for key in BREAKDOWN_KEYS:
        vals = [r["reward_info"][key] for r in bought
                if isinstance(r.get("reward_info"), dict) and key in r["reward_info"]]
        s[f"mean_{key} (n={len(vals)})"] = float(mean(float(v) for v in vals)) if vals else None
    return s, rewards, steps


def plot(records, rewards, steps, out_png: Path):
    ok = [r for r in records if not r.get("error")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(rewards, bins=10, range=(0, 1), edgecolor="black")
    axes[0].set(title="Reward distribution", xlabel="reward", ylabel="episodes")

    axes[1].hist(steps, bins=range(0, max(steps, default=1) + 2), edgecolor="black")
    axes[1].set(title="Episode length", xlabel="steps", ylabel="episodes")

    colors = ["tab:green" if r.get("done") else "tab:red" for r in ok]
    axes[2].scatter(steps, rewards, c=colors, alpha=0.6)
    axes[2].set(title="Reward vs steps (green=bought, red=timed out)",
                xlabel="steps", ylabel="reward")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved plot to {out_png}")


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit("usage: python -m src.evaluation.summarize <episodes.jsonl> [...]")
    records = load(paths)
    s, rewards, steps = summarize(records)

    print("\nBaseline summary")
    print("================")
    for k, v in s.items():
        print(f"{k:38s} {v:.3f}" if isinstance(v, float) else f"{k:38s} {v}")

    stem = paths[0].with_suffix("")
    stem.with_name(stem.name + "_summary.json").write_text(json.dumps(s, indent=2))
    plot(records, rewards, steps, stem.with_name(stem.name + "_plots.png"))


if __name__ == "__main__":
    main()
