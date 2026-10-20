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

### Ollama

The baseline agent in `src/environment/main.py` calls a local model through [Ollama](https://ollama.com/download).

```bash
brew install ollama        # macOS; on Linux see https://ollama.com/download
ollama serve               # skip if the Ollama app is already running
ollama pull llama3:8b      
```
