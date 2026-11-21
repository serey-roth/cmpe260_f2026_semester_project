#!/bin/bash
# Run several slices of goals against an Ollama server on the GPU,
# both in Apptainer containers.
#
# Run inside a SLURM job, from the repo root, with these set:
#   MODEL     Ollama model in ~/ollama-models, e.g. qwen3.5:9b
#   SPLIT     train, dev or test
#   STARTS    first goal index of each slice, e.g. "0 100 200"
#   N         goals per slice
#   THINKING  on (default) or off (passes --no-thinking)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${MODEL:?set MODEL}" "${SPLIT:?set SPLIT}" "${STARTS:?set STARTS}" "${N:?set N}"

PORT=$((20000 + SLURM_JOB_ID % 20000))             # own port, in case the node is shared
export OLLAMA_BASE_URL=http://127.0.0.1:$PORT/v1   # where the shopper sends requests; overrides .env
export JAVA_HOME=$HOME/jdk                         # pyserini's search engine
export PYTHONUNBUFFERED=1

echo "model=$MODEL split=$SPLIT starts=($STARTS) n=$N thinking=${THINKING:-on}"
nvidia-smi --query-gpu=name,memory.total --format=csv

LOG=hpc/logs/ollama_$SLURM_JOB_ID.log
# --env, because the Ollama image sets its own OLLAMA_HOST. One request slot per slice; keep the model loaded
apptainer exec --nv --env OLLAMA_HOST=127.0.0.1:$PORT --env OLLAMA_MODELS=$HOME/ollama-models \
  --env OLLAMA_NUM_PARALLEL=$(wc -w <<< "$STARTS") --env OLLAMA_KEEP_ALIVE=-1 \
  ~/containers/ollama.sif ollama serve > "$LOG" 2>&1 &
# Ready once it answers and has logged which devices it found (it answers a few seconds before that)
timeout 120 sh -c "until curl -s http://127.0.0.1:$PORT > /dev/null && grep -q 'inference compute' $LOG; do sleep 2; done" \
  || { echo "Ollama did not start; see $LOG"; exit 1; }
# Ollama falls back to the CPU if it can't use the GPU, which makes runs many times slower
grep -qi "library=cuda" "$LOG" || { echo "Ollama found no GPU; see $LOG"; exit 1; }   # -i: logged as CUDA

pids=()
for start in $STARTS; do
  apptainer exec ~/containers/python.sif .venv/bin/python -m src.evaluation.baseline \
    --split "$SPLIT" --start "$start" --n "$N" --model "$MODEL" $([ "${THINKING:-on}" = off ] && echo --no-thinking) \
    > "hpc/logs/slice_${start}_$SLURM_JOB_ID.out" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"   # not plain wait: the Ollama server never exits
