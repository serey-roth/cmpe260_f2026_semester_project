# Single-Agent Baseline

Results and failure analysis for the single-agent WebShop baseline.

## Metrics

| Metric | Definition |
| --- | --- |
| Reward | Mean WebShop reward (0–1); episodes with no purchase score 0 |
| Success | Share of goals with reward = 1.0 |
| Purchase | Share of goals where the agent bought anything |
| Option | Mean `r_option` over purchases where the goal had options |

## Test: llama3:8b, Direct Prompt

500 goals, 15 max steps

| Reward | Success | Purchase | Avg steps |
| --- | --- | --- | --- |
| 0.545 | 14.8% | 96.6% | 3.6 |

| Reward bucket | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Episodes | 35 | 21 | 41 | 48 | 16 | 85 | 19 | 124 | 36 | 74 |

**Where the reward is lost:**

- Options skipped in 336 / 377 purchases that needed them (mean option score 0.07). Every one of these opened the first product and bought immediately.
- Exact target product bought in only 114 / 483 purchases. Attribute match is decent (`r_att` 0.70).
- Little exploration: about 1 search per episode; Description and Features were never opened.

## Dev: Model and Prompt Comparison

Same 50 dev goals. **Loops** = episodes that opened the same product 3+ times. **No buy** = episodes with no purchase. **No opt** = purchases without needed options.

| Model | Prompt | Steps | Reward | Success | Purchase | Option | Loops | No buy | No opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3:8b | CoT v1 | 15 | **0.483** | 16% | **76%** | 0.21 | 5 | 12 | 23 |
| qwen3.5:9b | direct v1 | 15 | 0.179 | 12% | 22% | **0.74** | 26 | 39 | 2 |
| qwen3.5:9b | CoT v1 | 15 | 0.266 | 16% | 40% | 0.59 | 10 | 31 | 5 |
| qwen3.5:9b | direct v2 | 25 | 0.242 | 18% | 30% | 0.64 | 28 | 35 | 4 |
| qwen3.5:9b | direct v3 | 25 | 0.245 | 16% | 34% | 0.60 | 30 | 33 | 5 |
| qwen3.5:9b | CoT v3 | 25 | 0.338 | **22%** | 46% | 0.58 | 19 | 27 | 7 |

Prompt versions:

- **v1:** original prompt, with the last 5 actions as history.
- **v2:** v1 plus "buy once the product fits; don't search again."
- **v3:** v2 plus "options reset when you leave a product page", with the full action history.

## Findings

1. **The models fail in opposite ways.** llama3 buys too early and skips options. qwen3.5 picks options well but often never buys.
2. **CoT helps qwen3.5:** +0.09 reward, +6 points of success and fewer loops than direct on the same setup (v3, 25 steps).
3. **Best success rate is qwen3.5 CoT at 22%.** llama3's higher reward comes from partial credit on near-miss purchases.
4. **The loops come from indecision, not memory.** Full history didn't reduce them. Qwen won't settle when a product almost fits (e.g. 5 oz instead of 6 oz).
5. **More steps mostly mean more looping.**
6. **Single-agent prompting has plateaued** at 12–22% success.

## Notes

- **Thinking tags:** the CoT prompt uses `<reasoning>` tags, because Ollama strips `<think>` from qwen3.x replies. Native thinking is off (`think=False`); it is very slow.
- **Ollama stalls:** the qwen3.5 runner stalls occasionally. Requests time out after 120 s and are retried up to 5 times. Interrupted runs continue with `--resume <log>`.
- **Speed:** a 50-goal qwen3.5 CoT run takes over an hour on a 32 GB Mac.
- **Comparability:** the llama3 CoT row used the v1 prompt and 15 steps.
- **Corrupted run file:** `runs/dev_qwen3.5-9b_cot_20261031_135202.jsonl` was overwritten by a later resume, and its log has goal 541 twice. The table uses the original 50-goal numbers.
