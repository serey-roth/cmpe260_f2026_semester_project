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

### Data

The sample data that comes with WebShop (from [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop)) is 1,000 products and gives only 13 human-written goals, which is not enough for RL training. The Google Drive links in WebShop's setup script for the full data are dead, so we:

1. download the entire catalog from Hugging Face (about 5.7 GB),
2. generate the train/dev/test splits from it,
3. build a search index for the catalog, because the indexes shipped with WebShop only cover 1,000 products, and point the environment at it (`build_env` in `src/environment/env.py`).

Refer to [src/data/README.md](src/data/README.md) for data setup.

### Ollama

The baseline agent in `src/agents/main.py` calls a local model through [Ollama](https://ollama.com/download).

```bash
brew install ollama        # macOS; on Linux see https://ollama.com/download
ollama serve               # skip if the Ollama app is already running
ollama pull llama3:8b      
```

## Run the baseline agent

After the data setup above, from the repo root:

```bash
uv run python -m src.agents.main
```

This builds the environment, runs the agent, and saves the episode as a `.jsonl` log in `src/agents/logs/`.
