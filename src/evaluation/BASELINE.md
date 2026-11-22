# Baseline: Base, Teacher and SFT Student

Single-agent WebShop shopper in two reply formats, for three models:

- **JSON mode:** a JSON reply per step, with the chain-of-thought prompt v5 (Appendix A).
- **Tool mode:** a `search` or `click` tool call per step, with the tool prompt (Appendix C).

Models:

- **Base:** Qwen3.5-0.8B.
- **Teacher:** Qwen3.5-9B, which generated the SFT data.
- **Student:** Qwen3.5-0.8B fine-tuned (SFT) on the teacher's successful episodes.

## 1. Setup

- **Environment:** WebShop with the full catalog (1,181,436 products) and our own search index; at most 25 actions per episode.
- **Goals:** test results use all 500 test goals (goals 0–499); dev results use the first 50 dev goals (goals 500–549); the teacher data uses train goals 0–599.
- **Serving:** Ollama on one A100 of the SJSU HPC, with each model's sampling settings from Ollama. Our imported models copy those of Ollama's `qwen3.5:0.8b-bf16`: temperature 1, top_k 20, top_p 0.95, presence penalty 1.5.
- **Reply, JSON mode:** JSON `{"reasoning", "action"}`, with `reasoning` required and first (structured output).
- **Reply, tool mode:** one call to `search(reasoning, keywords)` or `click(reasoning, value)`, with `reasoning` required and first. Only the first call of a reply is executed.
- **Thinking:** the model's native (hidden) thinking, set per run. Off sends `reasoning_effort: "none"`. Replies are capped at 8,192 tokens, thinking included.
- **Message layout, JSON mode:** the prompt and the action history are one system message; the current page is the user message. The teacher data was generated with an earlier layout (section 3.4).
- **Message layout, tool mode:** the prompt is the system message; the first user message is the request and the first page; each tool call's result is the next page. An episode is one conversation, in an Ollama context of 65,536 tokens.

| Metric | Definition |
| :---- | :---- |
| Reward | Mean WebShop reward (0–1), combining type, attributes, options and price; episodes with no purchase score 0 |
| Success | Share of episodes with reward 1.0 |
| Purchase | Share of episodes that ended with a purchase |
| Option | Mean option score (`r_option`) over purchases whose goal had options |
| Loops | Episodes that opened the same product 3 or more times |
| No buy | Episodes with no purchase |
| No opt | Purchases that needed options but had none selected |
| Invalid / ep | Mean number of actions per episode that the environment ignored: anything other than `search[<keywords>]` or `click[<value>]` with a value from the current page; in tool mode, also a reply without a readable tool call |

## 2. Results (test, 500 goals)

Each model was run once on all 500 test goals in each mode, after every choice (epoch, settings) was made on dev.

### 2.1 JSON mode

| Model | Thinking | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Teacher (Qwen3.5-9B) | on | 0.641 | 38.8% (194) | 86.8% (434) | 0.51 | 8.4 | 38 | 66 | 123 | 0.22 |
| Base (Qwen3.5-0.8B, our GGUF) | off | 0.347 | 14.6% (73) | 54.0% (270) | 0.18 | 16.9 | 273 | 230 | 129 | 4.04 |
| **Student (SFT, epoch 2)** | off | **0.666** | **39.8% (199)** | **91.6% (458)** | **0.53** | **7.8** | **6** | **42** | **98** | 0.27 |

| Model | Type | Attributes | Options | Price | Searches / ep | Products viewed / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Teacher | 0.94 | 0.78 | 0.51 | 0.99 | 1.98 | 1.09 |
| Base (our GGUF) | 0.92 | 0.78 | 0.18 | 0.99 | 6.64 | 1.51 |
| Student | 0.93 | 0.77 | 0.53 | 0.99 | 1.76 | 1.17 |

### 2.2 Tool mode

| Model | Thinking | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Teacher (Qwen3.5-9B)** | on | **0.699** | **46.2% (231)** | 91.4% (457) | **0.61** | 8.0 | 15 | 43 | **91** | **0.25** |
| Base (Qwen3.5-0.8B, our GGUF) | off | 0.073 | 1.4% (7) | 11.6% (58) | 0.00 | 23.4 | 118 | 442 | 47 | 7.98 |
| Student (tool SFT, epoch 2) | off | 0.683 | 41.4% (207) | **93.8% (469)** | 0.51 | **6.5** | **7** | **31** | 118 | 0.50 |

| Model | Type | Attributes | Options | Price | Searches / ep | Products viewed / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Teacher | 0.94 | 0.79 | 0.61 | 0.99 | 1.83 | 1.40 |
| Base (our GGUF) | 0.95 | 0.88 | 0.00 | 0.98 | 13.74 | 0.77 |
| Student | 0.93 | 0.79 | 0.51 | 0.98 | 1.73 | 1.09 |

In 750 of the base's 11,677 steps, the reply had no readable tool call; the teacher and the student had none.

## 3. Results (dev, 50 goals)

### 3.1 Base, teacher and student

| Model | Thinking | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Teacher (Qwen3.5-9B) | on | 0.554 | 32% | 78% | 0.44 | 9.8 | 8 | 11 | 15 | 0.10 |
| Qwen3.5-9B | off | 0.595 | 24% | 86% | 0.26 | 7.8 | 5 | 7 | 22 | 0.32 |
| Base (Qwen3.5-0.8B, Ollama's build) | off | 0.371 | 14% | 60% | 0.15 | 15.7 | 26 | 20 | 18 | 3.80 |
| Base (Qwen3.5-0.8B, our GGUF) | off | 0.277 | 10% | 50% | 0.25 | 17.3 | 31 | 25 | 11 | 3.92 |
| **Student (SFT, epoch 2)** | off | **0.710** | **40%** | **94%** | **0.51** | 8.3 | **2** | **3** | 13 | 0.36 |

"Our GGUF" is the base model exported to Ollama the same way as the student (section 4), so it is the base the student is compared with.

| Model | Type | Attributes | Options | Price | Searches / ep | Products viewed / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Teacher (thinking on) | 0.96 | 0.78 | 0.44 | 1.00 | 2.38 | 1.30 |
| Qwen3.5-9B (thinking off) | 0.98 | 0.83 | 0.26 | 1.00 | 1.92 | 1.10 |
| Base (Ollama's build) | 0.95 | 0.80 | 0.15 | 1.00 | 5.80 | 1.38 |
| Base (our GGUF) | 0.84 | 0.61 | 0.25 | 1.00 | 7.22 | 1.46 |
| Student | 0.98 | 0.84 | 0.51 | 0.98 | 1.70 | 1.34 |

Reward components are means over purchases.

### 3.2 Student epochs

| Epoch | Validation loss | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 0.770 | 0.700 | 32% | 98% | 0.52 | 5.8 | 0 | 1 | 11 | 0.28 |
| **2** | **0.754** | **0.710** | **40%** | 94% | 0.51 | 8.3 | 2 | 3 | 13 | 0.36 |
| 3 | 0.860 | 0.645 | 40% | 84% | 0.55 | 9.6 | 2 | 8 | 10 | 0.80 |

### 3.3 Importing models into Ollama

| Model | Import | Reward | Success | Purchase | Avg steps | No buy | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Base | GGUF chat template (JSON schema dropped) | 0.060 | 6% | 6% | 24.0 | 47 | 4.66 |
| Base | Ollama's Qwen3.5 renderer and parser | 0.277 | 10% | 50% | 17.3 | 25 | 3.92 |
| Student | GGUF chat template (JSON schema dropped) | 0.602 | 34% | 78% | 9.7 | 11 | 1.54 |
| Student | Ollama's Qwen3.5 renderer and parser | 0.710 | 40% | 94% | 8.3 | 3 | 0.36 |

### 3.4 Message layout

The first layout sent the prompt and the history as separate system messages, after an extra system message from Pydantic AI about `<system>` tags. Qwen3.5's chat template accepts only one system message, so the current layout merges them and turns Pydantic AI's message off. The prompt text is the same.

| Model | Layout | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Qwen3.5-9B, thinking off | first | 0.641 | 28% | 94% | 0.34 | 7.3 | 3 | 3 | 0.28 |
| Qwen3.5-9B, thinking off | current | 0.595 | 24% | 86% | 0.26 | 7.8 | 5 | 7 | 0.32 |
| Base (Ollama's build) | first | 0.280 | 12% | 48% | 0.14 | 18.3 | 33 | 26 | 5.08 |
| Base (Ollama's build) | current | 0.371 | 14% | 60% | 0.15 | 15.7 | 26 | 20 | 3.80 |

### 3.5 Base model failures

Invalid actions of the base (Ollama's build, first layout; 254 over 50 episodes, none from timeouts):

| Type | Share | Example |
| :---- | :---- | :---- |
| Value not on the page | 60% | prompt text (`click[Click values: back to search, ...]`), placeholders (`click[<keyword>]`), full product titles |
| Product ID not on the page | 26% | an ID from an earlier results page |
| Malformed action | 13% | `search[` with no keywords; rambling text inside the action |
| `click[search]` (not allowed) | 2% | `click[search]` |
| Buy now on a page without the button | 0.4% | `click[buy now]` |

With thinking on, the base's thinking ran past the 8,192-token reply cap on every step we inspected after its first search, and no episode finished.

### 3.6 Tool mode

| Model | Thinking | Reward | Success | Purchase | Option | Avg steps | Loops | No buy | No opt | Invalid / ep |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Teacher (Qwen3.5-9B) | on | 0.680 | 38% | 92% | 0.47 | 9.8 | 1 | 4 | 15 | 0.40 |
| Base (Qwen3.5-0.8B, our GGUF) | off | 0.034 | 0% | 8% | 0.00 | 24.0 | 10 | 46 | 4 | 8.14 |
| Student (tool SFT), epoch 1, run 1 | off | 0.650 | 26% | 98% | 0.37 | 6.0 | 2 | 1 | 21 | 0.70 |
| Student (tool SFT), epoch 1, run 2 | off | 0.578 | 18% | 92% | 0.28 | 6.2 | 1 | 4 | 23 | 0.48 |
| **Student (tool SFT), epoch 2** | off | **0.709** | **40%** | **98%** | 0.43 | **6.0** | **1** | **1** | 15 | 0.58 |
| Student (tool SFT), epoch 3 | off | 0.611 | 28% | 92% | 0.51 | 7.7 | 4 | 4 | 9 | 0.78 |

In 55 of the base's 1,198 steps, the reply had no readable tool call; the teacher and the students had none.

## 4. Teacher Data and SFT

| | |
| :---- | :---- |
| Teacher run | Qwen3.5-9B, thinking on, first layout; train goals 0–599 in 6 parallel slices on one A100 |
| Teacher result | 602 episodes (train goals 0–599, and 2 from the setup test): 239 successful (40%), reward 0.628, 515 purchases |
| SFT data | The 239 successful episodes, one example per valid step: 1,171 train and 210 validation examples (215 and 24 goals) |
| Prompts | Rebuilt by replaying each episode through the current agent code, so every prompt is what the agent sends at inference (current layout). The target is the teacher's logged reply, `{"reasoning", "action"}`; its native thinking was not logged |
| Training | `Qwen/Qwen3.5-0.8B`, text-only, full fine-tune with TRL `SFTTrainer`, loss on the reply only; 3 epochs (1,758 updates, batch 2, learning rate 2e-5), 1,905 s on one A100 |
| Tool SFT data | The same 239 successful episodes, replayed through the tool shopper: each logged step becomes a tool call with the teacher's reasoning and action, and the next page its result; one conversation per episode. 232 episodes convert (7 have a step with no valid logged reply): 210 train and 22 validation conversations. Tokens per conversation (Qwen3.5 chat template): median 3,107.5, max 23,951; 1,001,644 in all |
| Tool training | Same setup, on the tool data, with the loss on the tool calls only; 3 epochs. The training chat template keeps an empty thinking block on earlier assistant turns, which Ollama's renderer leaves out (4 tokens per turn) |
| Serving | Converted to GGUF with the llama.cpp release Ollama bundles, and imported into Ollama with Ollama's Qwen3.5 renderer and parser |

## 5. Findings

1. **SFT lifts Qwen3.5-0.8B from 14.6% to 39.8% success on test (JSON mode)** (0.347 to 0.666 reward; on dev, 10% to 40%). The student buys in 91.6% of episodes instead of 54.0%, loops in 6 instead of 273, makes 0.27 invalid actions per episode instead of 4.04, and searches 1.76 times per episode instead of 6.64.
2. **The student matches its teacher on test (JSON mode).** Against the teacher as it generated the data (thinking on): 39.8% against 38.8% success (199 against 194), 0.666 against 0.641 reward, and 0.53 against 0.51 option score. It buys more often (91.6% against 86.8%) and loops less (6 against 38 episodes). On dev the student was ahead (40% against 32% success); the test split does not confirm that lead. The student learned only from the teacher's successful episodes, and without its native thinking.
3. **Native thinking gives the teacher more successes and better options, but lower reward.** With thinking on, the 9B has 32% success against 24% and an option score of 0.44 against 0.26, but 0.554 reward against 0.595: it buys less often (78% against 86%) and loops more (8 against 5).
4. **Epoch 2 is the best student (JSON mode)**, by validation loss and by dev reward (0.710). With more epochs, episodes get longer (5.8, 8.3, 9.6 steps) and purchases fewer (98%, 94%, 84%); invalid actions per episode go from 0.36 (epoch 2) to 0.80 (epoch 3).
5. **An imported model needs Ollama's Qwen3.5 renderer and parser.** Without them, Ollama renders the prompt with the GGUF's chat template: the same prompt text, but the JSON schema is silently dropped, so replies can put the action before the reasoning. That cut the base to 6% success (3 purchases) and the student to 34%. With them, our GGUF gives the same replies as Ollama's build of the same weights (checked at temperature 0).
6. **The message layout changes little.** Going to one system message: the 9B (thinking off) from 28% to 24% success, the base from 12% to 14%.
7. **The base fails mostly on actions, not choices.** It makes 3.80 (current layout) and 5.08 (first layout) invalid actions per episode, 60% of them values not on the page, loops in 26 and 33 of 50 episodes, and cannot run with thinking on (3.5).
8. **In tool mode, the teacher leads on test.** 46.2% against 41.4% success (231 against 207), 0.699 against 0.683 reward. The gap is options: an option score of 0.61 against 0.51, and 91 against 118 purchases without the requested options. The student buys more often (93.8% against 91.4%) and loops less (7 against 15 episodes). On dev the student was ahead (40% against 38%).
9. **Tool mode helps the teacher most.** From JSON to tool mode on test, the teacher goes from 38.8% to 46.2% success (0.641 to 0.699 reward), the student from 39.8% to 41.4% (0.666 to 0.683). The tool student learned from the teacher's JSON-mode episodes, converted to tool calls.
10. **The base cannot use the tools without SFT.** On test it has 1.4% success and 11.6% purchases (JSON mode: 14.6% and 54.0%), 7.98 invalid actions and 13.74 searches per episode, and 750 steps with no readable tool call. Tool SFT lifts it to 41.4%.
11. **Dev differences on 50 goals are noisy.** The tool student's epoch 1, run twice on the same 50 goals, gave 26% and 18% success (0.650 and 0.578 reward). Epoch 2 was chosen by dev reward (0.709).

## Appendix A. Prompt (v5)

**System message**, rebuilt each step:

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

Before you select an action, compare the user request with the current page: Is this the right product? Which requested options are already selected and which are still missing? Is the price within the limit?
After you analyze the request, reply with EXACTLY one action, either 'search[<keywords>]' or 'click[<value>]' and your reasoning.
Keep the reasoning concise: at most 3 short sentences.

Example on a product page:
{"reasoning": "This is a black relaxed-fit sweatpant under $40, which matches. The request asks for black and xx-large. Neither is selected yet, so I select black first.", "action": "click[black]"}

Example on the search page:
{"reasoning": "I have not searched yet, so I search for the key words of the request.", "action": "search[xx-large black relaxed fit workout sweatpants]"}

Your actions so far:
{history}
You have {remaining_steps} actions left, and the task only counts if you finish by clicking buy now.
```

**User message**, each step:

```text
User request: {user_request}
Current page text: {observation}
Click values: {clickables}. {search_hint}
```

## Appendix B. Example Episode (Student, goal 500, reward 1.0)

**Request:** i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 80.00 dollars

| Step | Action | Reasoning |
| :---- | :---- | :---- |
| 1 | `search[small blue coated steel frame end table]` | I'm on the homepage with no products shown yet. I need to search for a small end table that matches the user's criteria: blue-coated steel frame, easy to assemble, under $80. |
| 2 | `click[b07yz35f4h]` | None of the visible products on this search results page mention 'blue' color or steel frame in a specific way. I need to check product details further by clicking ASINs that might match better criteria. The first ASIN B07YZ35F4H has Anti-Rust Coating and matches the easy assembly requirement. |
| 3 | `click[blue]` | This product matches: small end table with anti-rust coating at $53.99 (under $80). However, it's currently marked as Gold and I need Blue. I should select the blue color option first. |
| 4 | `click[Buy Now]` | The product matches all requirements: blue-coated steel frame, small end table type, and price under $80. Blue color is already selected. This is a valid match. |

## Appendix C. Tool Prompt

**System message:**

```text
You are a personal shopper browsing an ecommerce website. Find and buy the product that best matches the user's request.
You will be given the user's request and the current page's text.

The best match should satisfy these conditions:
1. It matches what the user asked for.
2. It has the features the user asked for. Check the title, and open Description or Features at most once per product. A feature counts as met unless the page contradicts it.
3. Every option the user asked for (such as color or size) is selected on the product page.
4. Its price is within any price limit in the request.

Use the following tools to find the best match:
- search: search the shop for products. Works on a page with a search bar.
- click: click a product ID, option, button or link on the current page.
Each tool returns the next page, its click values, and how many actions you have left. If the action is not available on the page, nothing happens and the tool says so.

Rules:
- On a search results page, compare titles and prices, then click the ASIN of the closest match. Titles rarely confirm every detail, so a missing detail is no reason to skip a product. Click next > only if no product is the right type within the price limit.
- On a product page, select the requested options. Once the product fits the request and its options are selected, click buy now.
- Options reset when you leave a product page. Only options shown as [clicked button] on the current page are selected.
- Description and Features pages have no buy now button. Click < prev to return to the product page; your selected options are kept.
- "back to search" abandons the product and loses your selected options. Only use it if this product does not fit the request.
- Do not reopen a product you already looked at UNLESS you are choosing between products.
- Buy the product as soon as you find the first one that fits the user request with the desired options are selected. DO NOT SEARCH AGAIN FOR SOMETHING BETTER.
- If no product matches exactly after you have looked at the results, open the closest one, select the closest options, and click buy now. Buying a close match is better than buying nothing.

Call exactly one tool per step. In its reasoning, compare the user request with the current page: Is this the right product? Which requested options are already selected and which are still missing? Is the price within the limit?
Keep the reasoning concise: at most 3 short sentences.
```

**Tools:**

| Tool | Description | Parameters |
| :---- | :---- | :---- |
| `search` | Search the shop for products. Works on a page with a search bar. | `reasoning`: why this action, at most 3 short sentences. `keywords`: what to search for. |
| `click` | Click a product ID, option, button or link on the current page. | `reasoning`: why this action, at most 3 short sentences. `value`: exactly one of the click values listed for the current page. |

**First user message**, then each tool result (after an action not on the page, the result starts with "Nothing happened: that action is not available on this page. Use search or click with a listed value."):

```text
User request: {user_request}            (first user message only)
Current page text: {observation}
Click values: {clickables}. {search_hint}
You have {remaining_steps} actions left, and the task only counts if you finish by clicking buy now.
```
