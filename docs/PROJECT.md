# Post-Training LLM with Reinforcement Learning for Long-Horizon Online Shopping

## Problem

Large Language Models (LLMs) are highly capable of handling general-purpose tasks; yet, they struggle on “long-horizon” tasks that require them to execute sequences of dependent actions over extended periods of time [1]. While multi-agent systems have recently become a popular framework for tackling these complex tasks, they introduce significant architectural overhead. Additionally, as task horizon increases, the overall system accuracy decreases due to compounding errors during agent-to-agent communication, causing workflows to lose track of their original goal.

Post-training with Reinforcement Learning (RL) provides an alternative and adaptive framework for optimizing a pre-trained LLM for a specific task. One common paradigm is Reinforcement Learning with Verifiable Rewards (RLVR), which leverages objective and verifiable criteria to assess the model’s generated output, rather than relying on human feedback. This approach is highly effective for tasks such as code generation, mathematical problem solving and e-commerce web navigation. While RLVR requires no explicit human supervision, they are prone to reward hacking, which is when agents learn to exploit the verification system rather than genuinely reaching the goal. [2].

Our project aims to investigate whether RL post-training yields a web agent that is more capable and efficient than a multi-agent system within the context of online web shopping. We further evaluate if the post-trained agent’s rewards reflect genuine task success and not shortcut behaviors.

## Data Set

We will use WebShop for both training and testing phases. WebShop is a simulated e-commerce website environment with 1.18 million real-world products and 12,087 crowd-sourced text instructions. Within this environment, an agent must navigate web pages to find, customize and purchase items according to specific natural-language-based  product requirements [3].

## Project Steps

We will implement the baseline multi-agent system using LangGraph, which allows us to orchestrate multiple stateful agents within a single workflow.

We will also divide the Webshop dataset into training and evaluation splits. The evaluation splits will be used as a benchmarking set for the LangGraph system and the post-trained RL framework. We will also create a unified verifier directly from the Webshop dataset for assessing task execution and reward hacking behaviors.

For post-training, we choose Group Relative Policy Optimization (GRPO) because it is suitable for rule-based tasks with objective verification, such as checking for correct purchased items. It also avoids training a separate critic model, making it computationally more efficient than Proximal Policy Optimization (PPO). We will use RAGEN’s GRPO training pipeline and its built-in WebShop environment. We will first perform Supervised Fine-Tuning (SFT) on a small set of trajectories, followed by GRPO training with WebShop's rule-based reward function.

## Expected Results

We will evaluate the multi-agent baseline, the SFT-only framework, and the SFT+GRPO framework on our benchmarking evaluation splits. For each configuration, we report the average reward, the success rate, the attribute/option/price/type breakdown, and the trajectory statistics (length, items viewed, searches). These results will be compared directly against Webshop’s rule baseline, prior best model and human performance for meaningful context. Lastly, we will test GRPO checkpoints throughout the evaluation process against our verifier to determine the degree of reward overoptimization and assess reward fidelity.

## References

- Evaluating Long-Context Reasoning in LLM-Based WebAgents
- [2604.15149] LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking
- [2207.01206] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
- Scaling Laws for Reward Model Overoptimization.
