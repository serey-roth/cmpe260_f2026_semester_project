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
