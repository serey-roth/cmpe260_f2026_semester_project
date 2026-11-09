# rl-webshop

Post-training a large language model with reinforcement learning for long-horizon online shopping on the [WebShop](https://github.com/ZihanWang314/webshop-minimal) environment. See the [project description](docs/PROJECT.md) for motivation and goals.

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and a JDK (used by `pyserini`), e.g. `sudo apt install default-jdk`, or on macOS `brew install openjdk` followed by the symlink step brew prints (`sudo ln -sfn $(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk`).

```bash
git clone --recurse-submodules https://github.com/serey-roth/cmpe260_f2026_semester_project.git rl-webshop
cd rl-webshop
# already cloned without submodules? run: git submodule update --init

uv sync
```

On macOS, set the JDK and certificate bundle before running commands that
import Pyserini:

```bash
export JAVA_HOME="$(/usr/libexec/java_home)"
export SSL_CERT_FILE="$(uv run python -c 'import certifi; print(certifi.where())')"
```

`JAVA_HOME` is required because Pyserini starts a JVM. `SSL_CERT_FILE` avoids
certificate errors when Pyserini loads its index metadata.

### Data

The default configuration uses the bundled 1,000-product dataset and
prebuilt search index from
[webshop-minimal](https://github.com/ZihanWang314/webshop-minimal). It does
not require a download, split generation, or index-building step.

The full-catalog setup remains available by passing a directory containing
`items_shuffle.json` and `items_ins_v2.json` to the data configuration helpers.

### Ollama

The baseline agent in `src/agents/main.py` calls a local model through [Ollama](https://ollama.com/download).

```bash
brew install ollama        # macOS; on Linux see https://ollama.com/download
ollama serve               # skip if the Ollama app is already running
ollama pull llama3:8b      
```

## Run the baseline agent

From the repo root:

```bash
uv run python -m src.evaluation.baseline --split test --n 1
```

This runs one bundled-dataset episode and saves the detailed episode log and
summary under `src/agents/logs/` and `src/evaluation/runs/`.

## Preliminary baseline results (local GPU-backed setup)

The following summary is from a 10-episode test batch run on the local machine
with the Ollama model `llama3.2:3b` and the direct prompt style.

| Date | Model | Split | Episodes | Avg reward | Success rate | Purchase rate | Avg steps | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-11-07 | `llama3.2:3b` | `test` | 10 | 0.1033 | 0.0000 | 0.3000 | 20.1 | Local GPU-backed Ollama inference on RTX 4070 |

The run finished with the following metrics:

```json
{
  "type": "baseline_single_agent",
  "model": "llama3.2:3b",
  "prompt": "direct",
  "split": "test",
  "num_goals": 10,
  "timestamp": "20261107_125609",
  "average_reward": 0.10333333333333332,
  "average_steps": 20.1,
  "success_rate": 0.0,
  "purchase_rate": 0.3,
  "option_score": 0.0
}
```

## Multi-model comparison on the same WebShop batch

The following comparison uses the same 3-goal test batch, direct prompt style,
and the same local GPU-backed Ollama setup for each model.

> Note: T5 and Flan are not available in the local Ollama catalog on this machine, so the family comparison here is limited to the models that were actually installed and benchmarked through the project’s Ollama interface.

| Model | Family | Episodes | Avg reward | Success rate | Purchase rate | Avg steps | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `llama3.2:3b` | Llama | 3 | 0.1111 | 0.0000 | 0.3333 | 19.0 | Small local Llama model |
| `llama3.1:8b` | Llama | 3 | 0.3333 | 0.3333 | 0.3333 | 22.7 | Larger Llama model did best in the earlier run |
| `gemma3:4b` | Gemma | 3 | 0.7778 | 0.3333 | 1.0000 | 4.7 | Best average reward on this fixed batch; very fast and bought consistently |
| `qwen2.5:3b` | Qwen | 3 | 0.2222 | 0.0000 | 0.3333 | 17.7 | Strong search behavior, but often stalled before buying |

On this fixed 3-goal batch, `gemma3:4b` achieved the best overall performance by a wide margin, with a higher average reward and much shorter average episode length than the Llama runs. `qwen2.5:3b` showed some promising browsing behavior but did not complete the task reliably enough to match the stronger Gemma and Llama results.

If you want a true T5/Flan comparison, the next step is to run those models through a Hugging Face or vLLM endpoint instead of the current local Ollama-only setup, because the default Ollama registry here did not include T5/Flan variants.

The detailed per-episode logs are saved under `src/evaluation/runs/`.
