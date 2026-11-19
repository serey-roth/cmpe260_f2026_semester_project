# Single-Agent Baseline

Results and failure analysis for the single-agent WebShop baseline, and the setup for generating SFT teacher data. Prompts and an example episode are in the appendices.

Configurations are named `<Model>-<Prompt type>-<Prompt version>`; for example, Qwen3.5-CoT-v4 is Qwen3.5 with the chain-of-thought prompt, version 4. Qwen3.5 is the 9B model unless a size is given.

## 1. Setup

- **Environment:** WebShop with the full catalog (1,181,436 products) and our own search index.
- **Splits:** test 500 goals, dev 998, train 10,538. The test run used all 500 test goals; dev comparisons use the first 50 dev goals (goals 500–549).
- **Models:** run locally through Ollama: llama3:8b, qwen3.5:9b and qwen3.5:0.8b (bf16).
- **Prompt types:** **direct** asks only for the action; **CoT** asks the model to compare the request with the page first (Is this the right product? Which requested options are selected and which are missing? Is the price within the limit?) and to give its reasoning with the action.

### Metrics

| Metric | Definition |
| :---- | :---- |
| Reward | Mean WebShop reward (0–1), combining type, attributes, options and price; episodes with no purchase score 0 |
| Success | Share of episodes with reward 1.0 |
| Purchase | Share of episodes that ended with a purchase |
| Option | Mean option score (`r_option`) over purchases whose goal had options |
| Loops | Episodes that opened the same product 3 or more times |
| No buy | Episodes with no purchase |
| No opt | Purchases that needed options but had none selected |
| Invalid / ep | Mean number of actions per episode that the environment ignored |

An action is **invalid** when the environment ignores it: anything other than `search[<keywords>]` or `click[<value>]` with a value from the current page. Steps where the model gave no usable reply (a timeout, or invalid output after the retries) are recorded as the action `invalid`.

## 2. Results

### 2.1 Test split: Llama3-direct-v1 (500 goals, 15 max steps)

| Reward | Success | Purchase | Avg steps |
| :---- | :---- | :---- | :---- |
| 0.545 | 14.8% | 96.6% | 3.6 |

| Reward (rounded to 0.1) | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Episodes | 41 | 15 | 41 | 48 | 16 | 85 | 19 | 124 | 37 | 74 |

No options were selected in 336 of the 377 purchases that needed them (option score 0.07); 329 of those 336 episodes were search, open one product, buy. The goal's target product was bought in 114 of 483 purchases.

### 2.2 Dev split: all configurations (first 50 dev goals)

| Configuration | Thinking | Max steps | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama3-CoT-v1 | – | 15 | 0.483 | 16% | 76% | 0.21 | 7.3 | 5 | 12 | 23 | n/a |
| Qwen3.5-direct-v1 | off | 15 | 0.179 | 12% | 22% | 0.74 | 13.9 | 26 | 39 | 2 | n/a |
| Qwen3.5-CoT-v1 | off | 15 | 0.266 | 16% | 40% | 0.59 | – | 10 | 31 | 5 | n/a |
| Qwen3.5-direct-v2 | off | 25 | 0.242 | 18% | 30% | 0.64 | 20.1 | 28 | 35 | 4 | n/a |
| Qwen3.5-direct-v3 | off | 25 | 0.245 | 16% | 34% | 0.60 | 21.0 | 30 | 33 | 5 | n/a |
| Qwen3.5-CoT-v3 | off | 25 | 0.338 | 22% | 46% | 0.58 | 16.9 | 19 | 27 | 7 | n/a |
| Qwen3.5-CoT-v4 | on | 25 | **0.532** | **30%** | 74% | 0.48 | 11.6 | 10 | 13 | 12 | 0.16 |
| Qwen3.5-0.8B-CoT-v5 | off | 25 | 0.280 | 12% | 48% | 0.14 | 18.3 | 33 | 26 | 13 | 5.08 |

**Thinking** is the model's native (hidden) thinking; see the notes in section 4. Invalid actions were logged from Qwen3.5-CoT-v4 onward. The Qwen3.5-CoT-v1 row uses the original 50-goal summary: its episode log was later appended to by a resumed run and holds goal 541 twice, so its average step count can't be recovered (computed from the log as is: reward 0.261, success 16%, purchase 39%).

### 2.3 Reward breakdown and exploration

Reward components are means over purchases.

| Configuration | Type | Attributes | Options | Price | Searches / ep | Products viewed / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama3-direct-v1 (test) | 0.92 | 0.70 | 0.07 | 0.99 | 1.06 | 1.02 |
| Llama3-CoT-v1 | 0.94 | 0.81 | 0.21 | 1.00 | 1.08 | 1.16 |
| Qwen3.5-direct-v3 | 0.89 | 0.88 | 0.60 | 1.00 | 5.18 | 1.24 |
| Qwen3.5-CoT-v4 | 0.96 | 0.78 | 0.48 | 1.00 | 2.78 | 1.56 |
| Qwen3.5-0.8B-CoT-v5 | 0.90 | 0.67 | 0.14 | 1.00 | 6.50 | 1.62 |

### 2.4 Invalid actions

**Qwen3.5, dev runs from Nov 6–8** (20 invalid steps across the four dev runs that logged them):

| Type | Count | Example |
| :---- | :---- | :---- |
| No valid reply (timeout, or invalid output after retries) | 11 | `invalid` |
| Malformed action | 4 | `click[>]}` instead of `click[next >]` |
| Value not on the page | 4 | `click[> next]` (the button is `next >`) |
| Buy now on a Description/Features page | 1 | `click[buy now]` instead of `click[< prev]` first |

**Qwen3.5-0.8B-CoT-v5** (254 invalid steps over 50 episodes, none from timeouts):

| Type | Share | Example |
| :---- | :---- | :---- |
| Value not on the page | 60% | prompt text (`click[Click values: back to search, ...]`), placeholders (`click[<keyword>]`), full product titles |
| Product ID not on the page | 26% | an ID from an earlier results page |
| Malformed action | 13% | `search[` with no keywords; rambling text inside the action |
| `click[search]` (not allowed) | 2% | `click[search]` |
| Buy now on a page without the button | 0.4% | `click[buy now]` |

With thinking on, the steps we inspected after Qwen3.5-0.8B's first search all hit the 8,192-token reply cap, and no episode finished before the run was cancelled.

### 2.5 Teacher data generation (SJSU HPC)

| | |
| :---- | :---- |
| Teacher | Qwen3.5-9B, CoT v5, thinking on |
| Goals | train goals 0–599, 6 parallel slices of 100 |
| Hardware | 1 × NVIDIA A100 40 GB, Ollama in an Apptainer container |
| Throughput | about 136 episodes per hour, estimated from the first 15 minutes (35 episodes); about 11× the 32 GB Mac |
| Request speed | about 2.75 s per request with one slice running: about 107 tokens/s generation, 2,600 tokens/s prompt processing |
| Memory | 13.9 GB peak (`sacct` MaxRSS) for the 2-goal test with one slice |

## 3. Findings

1. **The models fail in opposite ways.** Llama3 buys early and skips options (option score 0.07 on test, 0.21 on dev). Qwen3.5 picks options well (0.58–0.74 through v3) but often never buys (27–39 of 50 episodes with no purchase).
2. **CoT helps Qwen3.5.** On the same v3 prompt and 25 steps, CoT beats direct by 0.09 reward and 6 points of success, with 19 loops instead of 30.
3. **Full history did not reduce loops.** v3 added the full action history and the options-reset rule; loops went from 28 (direct-v2) to 30 (direct-v3), and reward stayed at 0.24.
4. **More steps did not bring purchases.** Going from 15 to 25 steps (direct-v1 to direct-v2, with the v2 buy rule) raised purchases from 22% to 30%, but 35 of 50 episodes still ended without a purchase.
5. **v4 with native thinking reached the best results:** 30% success and 0.532 reward, with no-buy episodes down from 27 (CoT-v3) to 13. The prompt and the thinking setting changed together, so their separate effects are not known. The option score fell from 0.58 to 0.48.
6. **Qwen3.5-0.8B reaches 12% success with thinking off**, the same as Qwen3.5-direct-v1. Most of its losses are invalid actions (5.08 per episode, none from timeouts) rather than shopping decisions. It also loops (33 of 50 episodes) and averages 6.5 searches per episode.
7. **Qwen3.5-0.8B cannot run with thinking on**: its thinking runs past the 8,192-token reply cap (2.4).

## 4. Notes

- **Reasoning tags (v1–v3):** Llama3-CoT-v1 reasons inside `<think>` tags; the Qwen3.5 CoT runs use `<reasoning>` tags, because Ollama strips `<think>` from Qwen replies.
- **Native thinking:** v1–v3 call Ollama with `think=False`, added before any Qwen3.5 run. From v4 (Pydantic AI) thinking was on, because Pydantic AI's `thinking=False` setting is not sent to Ollama; this was found later. Thinking is now controlled by `--no-thinking`, which sends `reasoning_effort: "none"`. Replies are capped at 8,192 tokens, thinking included.
- **Temperature:** 0 in the v1–v3 code; not set from v4.
- **Retries:** invalid replies are retried up to 5 times; v1–v3 also retried requests that timed out, while from v4 a timeout goes straight to the `invalid` fallback. Interrupted runs continue with `--resume <log>`.
- **Speed:** a 50-goal dev run on the 32 GB Mac took about 25 minutes with the Qwen3.5 direct prompt and about 7 hours with Qwen3.5-CoT-v4.
- **Comparability:** the test row and the dev rows use different goals; the prompt version and the thinking setting differ between rows (see Appendix A).

## Appendix A. Prompt Versions

| Version | Runs | Change from the previous version | Reply format | History shown | Max steps |
| :---- | :---- | :---- | :---- | :---- | :---- |
| v1 | Llama3-direct-v1 (test); Llama3-CoT-v1, Qwen3.5-direct-v1, Qwen3.5-CoT-v1 (dev) | Original prompt (B.1) | Free text; action extracted with a regular expression | Last 5 actions | 15 |
| v2 | Qwen3.5-direct-v2 (dev) | Added a rule to buy once the product fits and not search again. The exact v2 wording was not saved; v3 words it as "Buy the product as soon as the product fits and the desired options are selected. DO NOT SEARCH AGAIN FOR SOMETHING BETTER." | Free text | Last 5 actions | 25 |
| v3 | Qwen3.5-direct-v3, Qwen3.5-CoT-v3 (dev) | Added: "Options reset when you leave a product page. Only options shown as [clicked button] on the current page are selected." Full action history instead of the last 5 (B.2) | Free text | All actions | 25 |
| v4 | Qwen3.5-CoT-v4 (dev) | Rewritten for structured output; relaxed conditions and new navigation rules (B.3) | JSON `{"action", "reasoning"}`; reasoning optional | All actions | 25 |
| v5 | SFT data generation (train), Qwen3.5-0.8B-CoT-v5 (dev) | Reasoning required and written before the action; at most 3 sentences (B.4) | JSON `{"reasoning", "action"}` | All actions | 25 |

v1–v3 sent one prompt per step. From v4 onward, the agent uses Pydantic AI with structured output, and the system prompt, the action history and the current page are sent as separate messages.

## Appendix B. Prompts

### B.1 v1

The full prompt sent at every step. `{...}` fields are filled in per step.

```text
You are a personal shopper browsing an ecommerce website. Find and buy the product that best matches the user's request.

The best match MUST satisfy the following conditions:
1. It matches what the user asked for.
2. It has every feature the user asked for (check the title, and open Description or Features if the title does not confirm it).
3. Every option the user asked for (such as color or size) is selected on the product page.
4. Its price is within any price limit in the request.

How to proceed:
- On a search results page, compare titles and prices, then click the ASIN of the best match. Use next > only if nothing fits.
- On a product page, select the requested options. Once the product fits the request and its options are selected, click[buy now].
- "< prev" and "back to search" leave the current page. Only use them if this product does not fit the request.
- Do not reopen a product you already looked at unless you are choosing between products.

User request:
{user_request}

Current page:
{observations}

Valid actions on this page:
- {search_hint}
- click[<value>] where <value> is exactly one of: {clickables}

Your recent actions:
{recent_history}
You have {remaining_steps} actions left, and the task only counts if you finish by clicking buy now.

Reply with exactly one action and nothing else, and ALWAYS include the square brackets.
Examples: search[blue wireless headphones], click[b09q55jlcs], or click[buy now]
```

### B.2 v3

v1 with two added rules, the full action history ("Your actions so far"), and a reply format that depends on the prompt type. Only the changed "How to proceed" section is shown. v2 was not saved; it added the buy rule and kept the last 5 actions as history.

```text
How to proceed:
- On a search results page, compare titles and prices, then click the ASIN of the best match. Use next > only if nothing fits.
- On a product page, select the requested options. Once the product fits the request and its options are selected, click[buy now].
- Options reset when you leave a product page. Only options shown as [clicked button] on the current page are selected.
- "< prev" and "back to search" leave the current page. Only use them if this product does not fit the request.
- Do not reopen a product you already looked at UNLESS you are choosing between products.
- Buy the product as soon as the product fits and the desired options are selected. DO NOT SEARCH AGAIN FOR SOMETHING BETTER.
```

**Direct reply format**:

```text
Reply with exactly one action and nothing else, and ALWAYS include the square brackets.
Examples: search[blue wireless headphones], click[b09q55jlcs], or click[buy now]
```

**CoT reply format**:

```text
First reason briefly inside <reasoning></reasoning> tags, then write exactly one action on the line after </reasoning>.
In your reasoning, compare the user request with the current page: Is this the right product? Which requested options are already selected and which are still missing? Is the price within the limit?
The action must be written as: search[<keywords>] or click[<value>].

Example reply on a product page:
<reasoning>
This is a black relaxed-fit sweatpant under $40, which matches. The request asks for black and xx-large. Neither is selected yet, so I select black first.
</reasoning>
click[black]

Example reply on the search page:
<reasoning>
I have not searched yet, so I search for the key words of the request.
</reasoning>
search[xx-large black relaxed fit workout sweatpants]
```

### B.3 v4

**System prompt**:

```text
You are a personal shopper browsing an ecommerce website. Find and buy the product that best matches the user's request.
You will be given the user's request, the current page's text, and action options, where you can either search for a product or click on a link or button.

The best match should satisfy these conditions:
1. It matches what the user asked for.
2. It has the features the user asked for. Check the title, and open Description or Features at most once per product. A feature counts as met unless the page contradicts it.
3. Every option the user asked for (such as color or size) is selected on the product page.
4. Its price is within any price limit in the request.

Rules:
- On a search results page, compare titles and prices, then click the ASIN of the closest match. Titles rarely confirm every detail, so a missing detail is no reason to skip a product. Use next > only if no product is the right type within the price limit.
- On a product page, select the requested options. Once the product fits the request and its options are selected, click[buy now].
- Options reset when you leave a product page. Only options shown as [clicked button] on the current page are selected.
- Description and Features pages have no buy now button. Use click[< prev] to return to the product page; your selected options are kept.
- "back to search" abandons the product and loses your selected options. Only use it if this product does not fit the request.
- Do not reopen a product you already looked at UNLESS you are choosing between products.
- Buy the product as soon as you find the first one that fits the user request with the desired options are selected. DO NOT SEARCH AGAIN FOR SOMETHING BETTER.
- If no product matches exactly after you have looked at the results, open the closest one, select the closest options, and click[buy now]. Buying a close match is better than buying nothing.
```

**Action history** (a second system message, rebuilt each step)

```text
Your actions so far:
{history}
You have {remaining_steps} actions left, and the task only counts if you finish by clicking buy now.
```

**CoT reply format** (a third system message)

```text
Before you select an action, compare the user request with the current page: Is this the right product? Which requested options are already selected and which are still missing? Is the price within the limit?
After you analyze the request, reply with EXACTLY one action, either 'search[<keywords>]' or 'click[<value>]' and your reasoning.

Example on a product page:
{"reasoning": "This is a black relaxed-fit sweatpant under $40, which matches. The request asks for black and xx-large. Neither is selected yet, so I select black first.", "action": "click[black]"}

Example on the search page:
{"reasoning": "I have not searched yet, so I search for the key words of the request.", "action": "search[xx-large black relaxed fit workout sweatpants]"}
```

**User message** (each step)

```text
User request: {user_request}
Current page text: {observation}
Click values: {clickables}. {search_hint}
```

### B.4 v5

Same text as v4, with three changes:

1. The CoT reply format is merged into the system prompt, so the action history comes last.
2. The reply format adds: "Keep the reasoning concise: at most 3 short sentences."
3. The output schema requires `reasoning` and places it before `action`, so the model reasons before it picks the action. In v4, `reasoning` was optional and came after the action, and the model sometimes skipped it (steps 1 and 3 in Appendix C).

## Appendix C. Example Episode (Qwen3.5-CoT-v4, reward 1.0)

**Request:** i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars

| Step | Action | Reasoning |
| :---- | :---- | :---- |
| 1 | `search[heavy duty dust proof tempered glass iphone 13 pro max red black]` | (none) |
| 2 | `click[b09p572dp9]` | The product matches the user request: iPhone 13 Pro Max 6.7 inch case + 4 protectors (2 screen + 2 camera), heavy duty/wine red color at $18.99 which is under $50. |
| 3 | `click[redblack]` | (none) |
| 4 | `click[case+4 protectors]` | The product matches the user request (iPhone 13 Pro Max case under $50 at $18.99, and redblack color is already selected). The only missing requirement is the 'case+4 protectors' option which needs to be selected before clicking buy now. |
| 5 | `click[buy now]` | This product matches the user request: it's a heavy duty case with tempered glass (screen/camera protectors) for iPhone 13 Pro Max 6.7 inch. The color 'redblack' is selected as requested, and 'case+4 protectors' is selected (2 screen + 2 camera protector = 4). Price $18.99 is under $50 limit. All requested options are selected, so I should buy now. |

**What the model receives at step 4** (system prompt from B.3 omitted; `...` marks lines cut here)

```text
[system] Your actions so far:
search[heavy duty dust proof tempered glass iphone 13 pro max red black]
click[b09p572dp9]
click[redblack]
You have 22 actions left, and the task only counts if you finish by clicking buy now.

[user] User request: i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars
Current page text: You have clicked redblack.
Instruction:
i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars
[button] Back to Search [button_]
[button] < Prev [button_]
color
  [button] black [button_]
  [button] pinkblue [button_]
  [button] purpleblue [button_]
  [clicked button] redblack [clicked button_]
  ...
size
  [button] case+2 protectors [button_]
  [button] case+2 protectors+clip [button_]
  [button] case+4 protectors [button_]
GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shockproof Drop Protective Cover for iPhone 13 Pro Max 6.7 Inch (WineRed+Lens)
Price: $18.99
Rating: N.A.
[button] Description [button_]
[button] Features [button_]
[button] Reviews [button_]
[button] Buy Now [button_]
Click values: ... search is not allowed on this page.
```

The page is as logged. The list of click values was not logged in this run, so it is not shown.

**Reply** (the logged action and reasoning)

```json
{"action": "click[case+4 protectors]", "reasoning": "The product matches the user request (iPhone 13 Pro Max case under $50 at $18.99, and redblack color is already selected). The only missing requirement is the 'case+4 protectors' option which needs to be selected before clicking buy now."}
```
