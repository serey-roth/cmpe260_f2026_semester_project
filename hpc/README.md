# Running on the SJSU CoE HPC

Teacher data, SFT and evaluation run as SLURM jobs on the HPC's A100s, in Apptainer containers.

| File | What it does |
| --- | --- |
| `run_episodes.sh` | Starts Ollama on the job's GPU and runs slices of goals against it |
| `generate_teacher.sbatch` | Generate teacher data (Qwen3.5-9B) on train goals 0–599 (6 slices of 100) |
| `run_baseline.sbatch` | Run a model on the first 50 dev goals (2 slices of 25) |
| `run_test.sbatch` | Run a model on the 500 test goals (5 slices of 100) |
| `train_sft.sbatch` | Run SFT of the student model |

Two machines share your home directory, so files are the same on both:

| | `coe-hpc1` | HPC3 (`ssh coe-hpc3` from `coe-hpc1`) |
| --- | --- | --- |
| Internet | yes | no |
| Use it for | setup (downloads, `uv sync`) | submitting and checking jobs |

Jobs only see HPC3's queue: run `sbatch`, `squeue` and `sacct` there.

**Apptainer** runs Docker images without root, as a single `.sif` file, with your home directory and (with `--nv`) the GPU available inside. We use two images: `python.sif` (Python 3.13, for our code and `.venv`) and `ollama.sif` (the model server).

## 1. Setup (once)

**Copy the repo** from your computer (over the VPN). This includes the gitignored splits, search index and SFT data:

```bash
rsync -av --exclude .venv --exclude '__pycache__' --exclude 'src/agents/logs' --exclude 'src/data/webshop-data' \
  --exclude 'src/data/search-index/documents' ~/Desktop/projects/post-train/ <SJSU_ID>@coe-hpc1.sjsu.edu:~/rl-webshop/
```

**On `coe-hpc1`:** containers, Java, the Python environment, the catalog and the models:

```bash
module load apptainer/1.3.2
mkdir -p ~/containers ~/.apptainer/tmp ~/ollama-models
export APPTAINER_CACHEDIR=~/.apptainer APPTAINER_TMPDIR=~/.apptainer/tmp UV_PYTHON_PREFERENCE=only-system

# containers (the full Python image, not -slim: SFT's Triton kernels need its C compiler)
apptainer pull ~/containers/python.sif docker://ghcr.io/astral-sh/uv:python3.13-bookworm
apptainer pull ~/containers/ollama.sif docker://ollama/ollama:latest

# Java for pyserini's search engine
mkdir -p ~/jdk && curl -L https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse | tar xz -C ~/jdk --strip-components=1

# Python environment, built inside the container it runs in
cd ~/rl-webshop && rm -rf .venv
apptainer exec --unsquash ~/containers/python.sif uv sync

# WebShop catalog (5.6 GB) and the student's base model (into the Hugging Face cache)
apptainer exec --unsquash ~/containers/python.sif .venv/bin/hf download YWZBrandon/webshop-data \
    items_shuffle.json items_ins_v2.json items_human_ins.json --repo-type dataset --local-dir src/data/webshop-data
apptainer exec --unsquash ~/containers/python.sif .venv/bin/hf download Qwen/Qwen3.5-0.8B

# Ollama models (--env, because the image sets its own OLLAMA_HOST)
apptainer exec --unsquash --env OLLAMA_MODELS=$HOME/ollama-models --env OLLAMA_HOST=127.0.0.1:11500 ~/containers/ollama.sif sh -c '
  ollama serve > /dev/null 2>&1 & until ollama list > /dev/null 2>&1; do sleep 1; done
  ollama pull qwen3.5:9b; ollama pull qwen3.5:0.8b-bf16; ollama list; kill $!'
```

**llama.cpp's converter**, for importing checkpoints into Ollama (section 4). It must be the release the Ollama image bundles:

```bash
apptainer exec ~/containers/ollama.sif ollama --version                                  # on HPC3, e.g. 0.40.16
curl -s https://raw.githubusercontent.com/ollama/ollama/v<version>/LLAMA_CPP_VERSION      # on coe-hpc1, e.g. b12177
git clone --depth 1 --branch <release> https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
```

**Check, then test** (on HPC3; 2 goals, a few minutes):

```bash
cd ~/rl-webshop && mkdir -p hpc/logs
ls src/data/search-index/index | head -3     # Lucene files
wc -l src/data/splits.jsonl                  # 12,037
THINKING=off MODEL=qwen3.5:9b STARTS=0 N=2 sbatch hpc/run_baseline.sbatch
```

The job log (`hpc/logs/run-baseline-dev_<JOBID>.out`) lists the A100s; the slice log (`hpc/logs/slice_0_<JOBID>.out`) shows steps and `done=... reward=...`. Delete the test's episode log and summary (`src/agents/logs/` and `src/evaluation/runs/`, `dev_single_qwen3.5-9b_*`).

## 2. Jobs

```bash
sbatch hpc/<job>.sbatch                       # settings as variables in front: THINKING=off MODEL=... sbatch ...
squeue -u $USER                               # R = running, PD = waiting for a GPU
tail -f hpc/logs/slice_0_<JOBID>.out          # episode progress
scancel <JOBID>
```

Jobs keep running when you log out. Model names are Ollama's, with a colon (`qwen3.5:9b`); results are written to `src/agents/logs/` (episodes) and `src/evaluation/runs/` (one summary per slice). Copy them to your computer with `rsync`, e.g.:

```bash
rsync -av '<SJSU_ID>@coe-hpc1.sjsu.edu:rl-webshop/src/agents/logs/test_single_*' src/agents/logs/
```

## 3. Teacher data and SFT

```bash
sbatch hpc/generate_teacher.sbatch                          # Qwen3.5-9B, thinking on, train goals 0–599
```

Build the SFT examples from the episodes (`python -m src.training.build_sft_data '<episode logs>'`), copy `data/sft/` to the HPC, then train:

```bash
sbatch hpc/train_sft.sbatch data/sft/<prompt_version> --max-steps 5     # test: the log should show the loss
rm -rf checkpoints/
sbatch hpc/train_sft.sbatch data/sft/<prompt_version>                   # 3 epochs
```

The student is saved to `checkpoints/qwen3.5-0.8b-sft-<prompt_version>/`, with a checkpoint per epoch (`checkpoint-586`, `-1172`, `-1758`).

## 4. Import a checkpoint into Ollama

Ollama on Linux can't import a safetensors checkpoint (it needs Ollama's MLX engine, which is Mac-only), so the export converts it to GGUF and writes a Modelfile. On HPC3, no GPU needed:

```bash
cd ~/rl-webshop
export HF_HUB_OFFLINE=1
apptainer exec ~/containers/python.sif .venv/bin/python -m src.training.export_qwen_checkpoints_for_ollama \
    checkpoints/qwen3.5-0.8b-sft-37b895a8/checkpoint-1172 checkpoints/ollama/qwen3.5-0.8b-sft-e2 --llama-cpp ~/llama.cpp

apptainer exec --env OLLAMA_MODELS=$HOME/ollama-models --env OLLAMA_HOST=127.0.0.1:11500 ~/containers/ollama.sif sh -c '
  ollama serve > /dev/null 2>&1 & until ollama list > /dev/null 2>&1; do sleep 1; done
  ollama create qwen3.5-0.8b-sft-e2 -f checkpoints/ollama/qwen3.5-0.8b-sft-e2/Modelfile
  ollama show qwen3.5-0.8b-sft-e2 | grep "template source"; kill $!'     # must say renderer_parser
```

Export the base model the same way (`Qwen/Qwen3.5-0.8B` as `qwen3.5-0.8b-text`), so base and student differ only in training. `checkpoints/ollama/` can be deleted after importing.

## 5. Evaluate

Thinking off for the 0.8B models (the student was trained without it), on for the teacher as it generated the data:

```bash
THINKING=off MODEL=qwen3.5-0.8b-sft-e2 sbatch hpc/run_baseline.sbatch 
THINKING=off MODEL=qwen3.5-0.8b-sft-e2 sbatch hpc/run_test.sbatch
THINKING=on MODEL=qwen3.5:9b sbatch hpc/run_test.sbatch
```

Use dev for every choice (which epoch, which settings), and the test split only once per final model.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Ollama found no GPU` | The job's GPU is unusable (cs002 was, so jobs exclude it). Resubmit; if a node keeps failing, add it to `--exclude` |
| `model "..." not found`, or every step `invalid` | The model name is wrong; check `ollama list` (names use a colon) |
| `ollama create` fails with `MLX runtime is not available` | A safetensors folder was imported directly; import the exported `Modelfile` instead |
| An imported model gives worse results, `template source` is `gguf_chat_template` | Its Modelfile lacks `RENDERER qwen3.5` / `PARSER qwen3.5`, so Ollama drops the JSON schema; re-export |
| `tensor ... not found` when loading an imported model | llama.cpp was not the release Ollama bundles; re-clone and re-export |
| `squeue` shows nothing | You're on `coe-hpc1`; jobs are on HPC3 |
| `xattr ... ENOTSUP` warnings during `apptainer pull` | Harmless |
