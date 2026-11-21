#!/bin/bash
# Run SFT of the student model on the GPU.
# Submit from HPC3 (g17): sbatch hpc/train_sft.sbatch data/sft/<prompt_version> [--max-steps 5]

#SBATCH --job-name=train-sft
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1                    # A100 nodes cs001-cs004; the P100s are too old for CUDA 13
#SBATCH --exclude=cs002                      # its GPUs give CUDA error 999 to new processes (Nov 2026)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
apptainer exec --nv --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 ~/containers/python.sif \
  .venv/bin/python -m src.training.train_sft "$@"
