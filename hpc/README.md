# Teacher data generation on the SJSU CoE HPC

Two SLURM jobs share one script:

| File | What it does |
| --- | --- |
| `run_episodes.sh` | Starts Ollama on the job's GPU and runs several slices of goals against it. Everything runs in Apptainer containers, because the HPC's glibc (2.17) is too old for the torch and faiss wheels. |
| `generate_teacher.sbatch` | Teacher data: Qwen3.5-9B on train goals 0–599 (6 slices of 100), 7-day queue |
| `run_baseline.sbatch` | Dev baseline of any model on the first 50 dev goals (2 slices of 25), 4-hour queue |

Settings (`MODEL`, `SPLIT`, `STARTS`, `N`, `THINKING`) can be overridden when submitting, e.g. `THINKING=off MODEL=qwen3.5:9b sbatch hpc/run_baseline.sbatch`.

Two machines share your home directory:

| | `coe-hpc1` | HPC3 (`g17`, `ssh coe-hpc3` from `coe-hpc1`) |
| --- | --- | --- |
| Internet | yes | no |
| Use it for | one-time setup (downloads) | submitting jobs (`gpuqs`, `gpuqm`) |

## One-time setup on `coe-hpc1`

First copy the repo from the Mac (over the VPN). This includes our `splits.jsonl` and search index, which are gitignored and copied rather than rebuilt so goals and search results match the Mac evaluations:

```bash
rsync -av --progress \
  --exclude .venv --exclude models --exclude '__pycache__' --exclude .DS_Store \
  --exclude 'src/agents/logs' --exclude 'src/data/webshop-data' --exclude 'src/data/search-index/documents' \
  ~/Desktop/projects/post-train/ <SJSU_ID>@coe-hpc1.sjsu.edu:~/rl-webshop/
```

**On Windows.** Everything after this copy runs on the HPC, so it is the same from any computer; `ssh` is built into PowerShell (or use PuTTY). Only the two copy steps (this one and copying the episodes back) need a different tool:

- **WSL** (Ubuntu on Windows) has `rsync`, so the commands above work as they are, with a Windows path such as `/mnt/c/Users/<name>/rl-webshop/`.
- **PowerShell `scp`** works too, but cannot skip folders: remove `.venv`, `src/data/webshop-data` and `src/agents/logs` from the copy first, then run `scp -r C:\path\to\rl-webshop <SJSU_ID>@coe-hpc1.sjsu.edu:~/`.

**1. Containers.** Use Apptainer (the successor of Singularity, same commands); check it is also available on HPC3: `module avail 2>&1 | grep -i -E "singularity|apptainer"`.

```bash
module load apptainer/1.3.2
mkdir -p ~/containers ~/.apptainer/tmp
export APPTAINER_CACHEDIR=~/.apptainer APPTAINER_TMPDIR=~/.apptainer/tmp   # /tmp may be too small
apptainer pull ~/containers/python.sif docker://ghcr.io/astral-sh/uv:python3.13-bookworm-slim
apptainer pull ~/containers/ollama.sif docker://ollama/ollama:latest
```

`coe-hpc1`'s kernel is too old to mount images directly, so steps 3–5 use `--unsquash`, which unpacks the image to `APPTAINER_TMPDIR` for each run. The job on HPC3 doesn't need it.

**2. Java for pyserini** (a plain JDK works both on the host and in the container):

```bash
mkdir -p ~/jdk && curl -L https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse | tar xz -C ~/jdk --strip-components=1
```

**3. Python environment, built inside the container** so it matches where it runs:

```bash
cd ~/rl-webshop
rm -rf .venv                                        # an environment built outside the container won't work
export UV_PYTHON_PREFERENCE=only-system             # use the container's Python 3.13
apptainer exec --unsquash ~/containers/python.sif uv sync
```

**4. Catalog** (public WebShop data, about 5.6 GB):

```bash
apptainer exec --unsquash ~/containers/python.sif uv run hf download YWZBrandon/webshop-data \
    items_shuffle.json items_ins_v2.json items_human_ins.json \
    --repo-type dataset --local-dir src/data/webshop-data
```

**5. Model** (pulled once into `~/ollama-models`):

```bash
mkdir -p ~/ollama-models
# --env, because the Ollama image sets its own OLLAMA_HOST; one run, so the image is unpacked once
apptainer exec --unsquash --env OLLAMA_MODELS=$HOME/ollama-models --env OLLAMA_HOST=127.0.0.1:11500 \
    ~/containers/ollama.sif sh -c \
    'ollama serve > ~/ollama-pull.log 2>&1 & until ollama list > /dev/null 2>&1; do sleep 1; done; ollama pull qwen3.5:9b; ollama list; kill $!'
ls ~/ollama-models/manifests/registry.ollama.ai/library/   # should list qwen3.5
```

**Check:**

```bash
ls src/data/search-index/index | head -3     # Lucene files
ls -lh src/data/webshop-data/                # items_shuffle.json ~5.4 GB
wc -l src/data/splits.jsonl                  # 12,037
```

## Test on HPC3

Two goals from far outside the real range, so the test doesn't overlap the real run:

```bash
ssh coe-hpc3
cd ~/rl-webshop && mkdir -p hpc/logs
srun -p gpuqs --gres=gpu:1 --exclude=g[1,3-5,9,11-12,14-16] --cpus-per-task=8 --mem=64G --time=2:00:00 --pty /bin/bash
STARTS="10000" N=2 bash hpc/generate_teacher.sbatch
```

## Run

From HPC3:

```bash
cd ~/rl-webshop && mkdir -p hpc/logs
sbatch hpc/generate_teacher.sbatch           # Submitted batch job <JOBID>
squeue -u $USER                              # R = running, PD = waiting
tail -f hpc/logs/slice_0_<JOBID>.out
wc -l src/agents/logs/train_single_*.jsonl   # episodes per slice
scancel <JOBID>                              # stop it
```

Copy the episodes back to the Mac:

```bash
rsync -av <SJSU_ID>@coe-hpc1.sjsu.edu:~/rl-webshop/src/agents/logs/train_single_* ~/Desktop/projects/post-train/src/agents/logs/
```
