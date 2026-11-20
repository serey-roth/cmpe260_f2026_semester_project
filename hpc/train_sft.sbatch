#!/bin/bash
# SFT of the student (Qwen/Qwen3.5-0.8B by default) on one GPU, with src.training.train_sft.
# The model must already be in the Hugging Face cache (see hpc/README.md, "SFT"): compute nodes have no internet.
# Submit from HPC3 (g17): DATA=data/sft/<prompt_version> sbatch hpc/train_sft.sbatch

#SBATCH --job-name=train-sft
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=g[1,3-5,9,11-12,14-16]   # P100 nodes: no bf16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${DATA:?set DATA=data/sft/<prompt_version>}"
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
PYTHON_SIF=${PYTHON_SIF:-$HOME/containers/python.sif}

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
SINGULARITY=$(command -v apptainer || command -v singularity)

export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

echo "model=$MODEL data=$DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$SINGULARITY" exec --nv "$PYTHON_SIF" .venv/bin/python -m src.training.train_sft "$DATA" --model "$MODEL" "$@"
