# rl-webshop

Post-training a large language model with reinforcement learning for long-horizon online shopping on the [WebShop](https://github.com/ZihanWang314/webshop-minimal) environment. See the [project description](docs/PROJECT.md) for motivation and goals.

## Setup

The project currently targets Python 3.13 or newer and uses the
`webshop-minimal` implementation at commit
`4758174947e5494f0cda3b762508ed4a20a2e4cf`. A JDK is required by
Pyserini's search backend.

### macOS setup

The following steps use the Conda environment used to validate this project.
If the environment already exists, skip the `conda create` command.

```bash
cd cmpe260_f2026_semester_project

conda create -n sklearn-env python=3.14 -y
conda activate sklearn-env

# The exact dependency source expected by pyproject.toml.
git clone https://github.com/ZihanWang314/webshop-minimal.git webshop
cd webshop
git checkout 4758174947e5494f0cda3b762508ed4a20a2e4cf
cd ..

python -m pip install --no-deps --no-build-isolation -e ./webshop
python -m pip install -e .
python -m pip install pytest requests-mock
```

Install Java if it is not already available:

```bash
brew install openjdk
```

Confirm Java is available before running the program:

```bash
java -version
export JAVA_HOME=$(/usr/libexec/java_home)
```

On Apple Silicon macOS, Torch and Pyserini can load different OpenMP
runtimes. If the program aborts with an `OMP: Error #15` message, set this
compatibility variable for the current shell:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Linux setup

Install Python 3.13+, Java, and `uv`, then install the project dependencies:

```bash
git clone --recurse-submodules https://github.com/serey-roth/cmpe260_f2026_semester_project.git rl-webshop
cd rl-webshop

git clone https://github.com/ZihanWang314/webshop-minimal.git webshop
cd webshop
git checkout 4758174947e5494f0cda3b762508ed4a20a2e4cf
cd ..

uv sync
```

If the project was cloned without submodules, initialize them with:

```bash
git submodule update --init
```

### Verify the installation

From the project root, verify that the application imports and run the
available tests:

```bash
python -c "import webshop_minimal; import src.environment.main; print('main import: OK')"
python -m pytest third_party/webshop/tests
```

The bundled tests belong to the separate Princeton WebShop source under
`third_party/webshop`; the application itself uses the `webshop-minimal`
package installed from `./webshop`.

## Evaluation tests

Fast, offline unit tests for the baseline agent are in
`src/evaluation/test_agent.py`. They test action parsing and step formatting
without starting WebShop or making Ollama requests:

```bash
python -m pytest src/evaluation/test_agent.py
```

These tests should normally take **about 1-3 seconds**. The first run in a
fresh Python process may take a few additional seconds while Torch, Pyserini,
and the WebShop package are imported.

For a small environment smoke test, use the Python interpreter after setting
the Java and macOS variables above:

```bash
python -c "from webshop_minimal import WebAgentTextEnv, init_basedir; init_basedir('small'); env = WebAgentTextEnv(observation_mode='text_rich', human_goals=True); env.reset(session=0); print('environment reset: OK'); print('clickables:', len(env.get_available_actions()['clickables']))"
```

The environment smoke test does not call Ollama and should take **about
2-10 seconds**, depending primarily on Pyserini and local disk speed.

An Ollama-backed evaluation should be treated as an integration test rather
than a unit test. One model action usually takes roughly **5-30 seconds** on
Apple Silicon, so a one-action smoke test is typically **10-45 seconds**
including startup. A complete 15-action episode can take **1-5 minutes**,
with the first run sometimes adding **10-60 seconds** while the model loads.

### Ollama

The baseline agent in `src/environment/main.py` calls a local model through [Ollama](https://ollama.com/download).

```bash
brew install ollama        # macOS; on Linux see https://ollama.com/download
ollama serve               # skip if the Ollama app is already running
ollama pull llama3:8b
```

## Running the baseline agent

After activating the environment and setting the macOS variables described
above, run one episode with:

```bash
python src/environment/main.py
```

The script initializes the bundled `small` dataset, runs goal `0`, allows at
most 15 actions, and writes one JSONL episode log under
`src/environment/logs/`. It requires Ollama to be running locally with the
`llama3:8b` model available.

## Expected runtime

One run is normally expected to take **about 1-5 minutes** on a modern
Apple-Silicon Mac. The main variable is the time Ollama takes to generate
each action:

- The first run can take an additional **10-60 seconds** while
  `llama3:8b` loads into memory.
- A successful episode may finish in only a few actions and take **under a
  minute**.
- An episode that uses the full 15-action limit can take **several minutes**.
- Lower-memory systems, CPU-only inference, or a busy Ollama server can take
  **5-10 minutes or longer**.

The program does not have a fixed wall-clock timeout. Its practical maximum is
the time required for 15 Ollama requests plus environment processing. To
measure a run in the current shell:

```bash
time python src/environment/main.py
```
