"""Summary metrics over episode records, shared by baseline.py and summarize_runs.py."""
from collections import Counter


def mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict]) -> dict:
    n = len(records)
    purchases = [r for r in records if r["done"]]
    option_scores = [r["reward_info"]["r_option"] for r in purchases if "r_option" in (r["reward_info"] or {})]
    viewed = [Counter(s["details"]["asin"] for s in r["steps"] if s["details"]["type"] == "product") for r in records]
    return {
        "num_goals": n,
        "average_reward": sum(r["reward"] for r in records) / n,
        "average_steps": sum(r["num_steps"] for r in records) / n,
        "success_rate": sum(r["reward"] == 1.0 for r in records) / n,  # bought exactly the right product
        "purchase_rate": sum(bool(r["done"]) for r in records) / n,  # bought anything
        "option_score": mean(option_scores),  # over purchases that had options to pick
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
