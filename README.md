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

`uv sync` installs WebShop's dependencies, not WebShop itself. It stays a submodule in `src/webshop/`, so put it on the path when running code that imports `webshop_minimal`:

```bash
PYTHONPATH=src/webshop uv run python -c "import webshop_minimal; print('ok')"
```
