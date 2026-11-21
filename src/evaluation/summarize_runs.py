"""Print BASELINE.md's table rows for runs, pooling each run's slices from its episode logs.

python -m src.evaluation.summarize_runs '<episode logs of run 1>' '<episode logs of run 2>' ...
e.g. 'src/agents/logs/dev_single_qwen3.5-0.8b-sft-e2_37b895a8_nothink_20261121_1348*.jsonl'

Pooling the episodes matters: the option score and the reward breakdown are means over purchases, so they can't be
averaged from the per-slice summaries.
"""
import glob
import json
import sys

from src.evaluation.metrics import summarize


def fmt(value: float | None, digits: int) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    print("| Run | Goals | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep "
          "| Type | Attributes | Options | Price | Searches / ep | Products viewed / ep |")
    for pattern in sys.argv[1:]:
        paths = sorted(glob.glob(pattern))
        records = [json.loads(line) for path in paths for line in open(path, encoding="utf-8")]
        assert records, f"no episodes match {pattern}"
        assert len({r["goal_idx"] for r in records}) == len(records), f"{pattern} has a goal more than once"
        m = summarize(records)
        b = m["reward_breakdown"]
        print(f"| {pattern} | {m['num_goals']} | {fmt(m['average_reward'], 3)} | {m['success_rate']:.0%} | {m['purchase_rate']:.0%} "
              f"| {fmt(m['option_score'], 2)} | {m['average_steps']:.1f} | {m['loops']} | {m['no_buy']} | {m['no_option_purchases']} "
              f"| {m['invalid_actions_per_episode']:.2f} | {fmt(b['r_type'], 2)} | {fmt(b['r_att'], 2)} | {fmt(b['r_option'], 2)} "
              f"| {fmt(b['r_price'], 2)} | {m['avg_searches']:.2f} | {m['avg_products_viewed']:.2f} |")
