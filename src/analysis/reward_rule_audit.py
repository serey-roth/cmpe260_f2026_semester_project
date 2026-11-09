"""Audit WebShop's reward rules for false matches (reward-hacking groundwork).

Runs WebShop's own matching functions on hand-made test cases and reports where
the reward counts something as matched that a human would not. No Ollama and no
catalog needed; runs in seconds.

Usage (from repo root):
    uv run python -m src.analysis.reward_rule_audit
"""
from pathlib import Path

from webshop_minimal.goal import get_attribute_reward, get_option_reward

OUT = Path(__file__).parent / "results" / "reward_rule_audit.md"

# (goal option, purchased option, should a human count it as matching?)
OPTION_CASES = [
    ("6 ounce", "16 ounce", False),
    ("pack of 2", "pack of 12", False),
    ("1 count", "12 count", False),
    ("small", "x-small", False),
    ("large", "x-large", False),
    ("black", "black red", False),
    ("blue", "navy blue", False),
    ("king", "california king", False),
    # controls
    ("black", "black", True),
    ("medium", "medium", True),
    ("red", "black", False),
]

# (goal attribute, where the text appears in the product, product text, should it match?)
ATTRIBUTE_CASES = [
    ("red", "Title", "Reduced Fat Snack Crackers", False),
    ("long", "Title", "Belong Wall Art Print", False),
    ("cotton", "Description", "Made of polyester, not cotton.", False),
    ("machine washable", "Description", "Not machine washable. Dry clean only.", False),
    ("gluten free", "Description", "Contains wheat. Not gluten free.", False),
    # controls
    ("gluten free", "Title", "Gluten Free Almond Crackers", True),
    ("cotton", "Description", "100% cotton fabric.", True),
]


def blank_product():
    return {"Title": "", "BulletPoints": [], "Description": "", "Attributes": []}


def main() -> None:
    rows, false_matches, total_wrong = [], 0, 0

    lines = ["# WebShop reward rule audit", "",
             "Each case runs WebShop's own matching code. **FALSE MATCH** = the reward gives credit a human would not.", "",
             "## Options (fuzzy match, token_set_ratio > 85)", "",
             "| Goal option | Purchased option | Reward counts it | Should count | Verdict |",
             "| --- | --- | --- | --- | --- |"]
    for goal, bought, should in OPTION_CASES:
        r, _ = get_option_reward([bought], [goal])
        matched = r == 1.0
        verdict = "ok" if matched == should else ("FALSE MATCH" if matched else "missed match")
        if not should:
            total_wrong += 1
            false_matches += matched
        lines.append(f"| {goal} | {bought} | {'yes' if matched else 'no'} | {'yes' if should else 'no'} | {verdict} |")

    lines += ["", "## Attributes (substring match in title, bullet points or description)", "",
              "| Goal attribute | Field | Product text | Reward counts it | Should count | Verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for attr, field, text, should in ATTRIBUTE_CASES:
        product = blank_product()
        product[field] = text
        r, _ = get_attribute_reward(product, {"attributes": [attr]})
        matched = r == 1.0
        verdict = "ok" if matched == should else ("FALSE MATCH" if matched else "missed match")
        if not should:
            total_wrong += 1
            false_matches += matched
        lines.append(f"| {attr} | {field} | {text} | {'yes' if matched else 'no'} | {'yes' if should else 'no'} | {verdict} |")

    lines += ["", f"**{false_matches} of {total_wrong} cases that should not match were counted as matches.**", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
