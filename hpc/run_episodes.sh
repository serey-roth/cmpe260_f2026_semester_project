#!/bin/bash
# Starts one Ollama server on the job's GPU and runs several slices of
# goals against it with src.evaluation.baseline. Both run in Apptainer containers, because the HPC's
# glibc (2.17) is too old for the torch and faiss wheels. One-time setup is in hpc/README.md.
#
# Run inside a SLURM job (sbatch or srun), from the repo root, with these set:
#   MODEL     Ollama model, e.g. qwen3.5:9b (must be pulled into ~/ollama-models)
#   SPLIT     train, dev or test
#   STARTS    first goal index of each slice, e.g. "0 100 200"
#   N         goals per slice
#   THINKING  on (default) or off (passes --no-thinking)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${MODEL:?set MODEL}" "${SPLIT:?set SPLIT}" "${STARTS:?set STARTS}" "${N:?set N}"
THINKING=${THINKING:-on}
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

echo "model=$MODEL split=$SPLIT starts=($STARTS) n=$N thinking=$THINKING"
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
    --split "$SPLIT" --start "$start" --n "$N" --model "$MODEL" $([ "$THINKING" = off ] && echo --no-thinking) \
    > "hpc/logs/slice_${start}_$SLURM_JOB_ID.out" 2>&1 &
  pids+=($!)
done
echo "started ${#pids[@]} slices: $STARTS"
wait "${pids[@]}"
