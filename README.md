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

## Environment Setup

WebShop's pinned dependencies were released for Python 3.8. Create an isolated
Conda environment instead of installing them into the system Python. The
commands below work on macOS and Linux:

```bash
conda create -n webshop python=3.8.13 -y
conda activate webshop
python --version
```

The last command should report Python 3.8.x. Conda may select a newer 3.8
patch release such as 3.8.20. From the project root, install
spaCy through Conda before installing the Python requirements:

```bash
cd third_party/webshop
conda install -c conda-forge spacy=3.3.0 pydantic=1.8.2 nmslib=2.1.1 gradio=3.16.2 -y
pip install -r requirements_arm.txt
pip install --no-deps pyserini==0.17.0
```

On Apple Silicon, set `PYTHONNOUSERSITE=1` before running WebShop commands if
your home directory contains packages installed for another architecture:

```bash
export PYTHONNOUSERSITE=1
```

On Apple Silicon (`arm64`), use `requirements_arm.txt`. It contains the
compatibility adjustments documented by WebShop, including NumPy 1.24.4 and
the newer Transformers and PyYAML versions. `nmslib`, `spacy`, and `gradio`
are installed through Conda because pip attempts incompatible source builds.
`pyserini` is installed without dependency resolution because its broad spaCy
requirement otherwise tries to replace the working Python 3.8 spaCy build.
The ARM setup script does not download the optional spaCy model because the
WebShop text environment does not require it.
On Intel macOS or Linux, use `requirements.txt` instead:

```bash
pip install -r requirements.txt
```

Install the remaining WebShop environment dependencies:

```bash
conda install -c pytorch faiss-cpu -y
conda install -c conda-forge openjdk=11 -y
```

Check that the environment and core WebShop modules can be imported:

```bash
python - <<'PY'
import sys
import gym
import numpy
import torch
import transformers
import web_agent_site

print("Python:", sys.version)
print("Executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("Gym:", gym.__version__)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("WebShop import: OK")
PY
```

Finally, download the small WebShop dataset and build its local search index:

```bash
cd /path/to/cmpe260_f2026_semester_project
bash src/scripts/setup_webshop.sh
```

The wrapper chooses `setup_arm.sh` automatically on Apple Silicon and
`setup.sh` on Intel macOS/Linux. It can be run from any directory.
The setup exits with an error if the upstream WebShop Google Drive data cannot
be downloaded; in that case the smoke test cannot run until the dataset files
are placed in `third_party/webshop/data/`.

Run the smoke test from the project root to verify that WebShop can load the
dataset, reset the text environment, expose the search bar, and execute one
search action:

```bash
conda activate webshop
export PYTHONNOUSERSITE=1
pytest -q tests/test_webshop_smoke.py
```

The setup script downloads data and models and can take some time. To use the
environment later:

```bash
conda activate webshop
cd /path/to/cmpe260_f2026_semester_project/third_party/webshop
```

If `conda` is not found, install Miniconda or Anaconda first. Do not use the
system Python 3.13 for WebShop; several pinned packages do not provide
compatible builds for it.

## References
- Evaluating Long-Context Reasoning in LLM-Based WebAgents
- [2604.15149] LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking
- [2207.01206] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
- Scaling Laws for Reward Model Overoptimization.
