#!/bin/bash
# Generates teacher episodes for SFT on the SJSU CoE HPC: one Ollama server on the job's GPU,
# with several baseline slices of train goals running against it.
#
# Both run in Singularity containers, because the HPC's glibc (2.17) is too old for torch and faiss.
# One-time setup is in hpc/README.md. Submit from HPC3 (g17), where the gpuqs/gpuqm queues are:
#   sbatch hpc/generate_teacher.sbatch
# Change the slices: STARTS="0 100" N=50 sbatch hpc/generate_teacher.sbatch

#SBATCH --job-name=generate-teacher-data
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:1
#SBATCH --exclude=g[1,3-5,9,11-12,14-16]   # P100 nodes: too old and small for this model
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                           # each slice loads the full catalog into RAM
#SBATCH --time=7-00:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --mail-user=serey.roth@sjsu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

STARTS=${STARTS:-"0 100 200 300 400 500"}     # a subset of 600 goals, increase for more, training has 10,538 goals
N=${N:-100}                                    # 100 goals per slice
MODEL=${MODEL:-qwen3.5:9b}
PYTHON_SIF=${PYTHON_SIF:-$HOME/containers/python.sif}
OLLAMA_SIF=${OLLAMA_SIF:-$HOME/containers/ollama.sif}

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
SINGULARITY=$(command -v apptainer || command -v singularity)

export OLLAMA_MODELS=$HOME/ollama-models
export OLLAMA_NUM_PARALLEL=$(wc -w <<< "$STARTS")   # one request slot per slice
export OLLAMA_KEEP_ALIVE=-1                          # keep the model loaded for the whole job
PORT=$((20000 + SLURM_JOB_ID % 20000))               # own port, in case the node is shared
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_BASE_URL=http://127.0.0.1:$PORT/v1     # overrides the value in .env
export JAVA_HOME=$HOME/jdk                           # pyserini's search engine
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv

# --env, because the Ollama image sets its own OLLAMA_HOST, which would win over the exported value
"$SINGULARITY" exec --nv \
  --env OLLAMA_HOST="$OLLAMA_HOST" --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  --env OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" --env OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" \
  "$OLLAMA_SIF" ollama serve > "hpc/logs/ollama_$SLURM_JOB_ID.log" 2>&1 &
for _ in $(seq 60); do curl -s "http://$OLLAMA_HOST" > /dev/null && break; sleep 2; done
curl -s "http://$OLLAMA_HOST" > /dev/null || { echo "Ollama did not start; see hpc/logs/ollama_$SLURM_JOB_ID.log"; exit 1; }

pids=()
for start in $STARTS; do
  "$SINGULARITY" exec "$PYTHON_SIF" .venv/bin/python -m src.evaluation.baseline \
    --split train --start "$start" --n "$N" --model "$MODEL" \
    > "hpc/logs/slice_${start}_$SLURM_JOB_ID.out" 2>&1 &
  pids+=($!)
done
echo "started ${#pids[@]} slices: $STARTS"
wait "${pids[@]}"
