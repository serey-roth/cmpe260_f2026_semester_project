#!/bin/bash
# Run a baseline of a model on the first 50 dev goals (2 slices of 25).
# Submit from HPC3 (g17): MODEL=qwen3.5:0.8b-bf16 sbatch hpc/run_baseline.sbatch

#SBATCH --job-name=run-baseline-dev
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1                    # A100 nodes cs001-cs004; the P100s are too old for CUDA 13
#SBATCH --exclude=cs002                      # its GPUs give CUDA error 999 to new processes (Nov 2026)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                            # about 14 GB per slice
#SBATCH --time=4:00:00
#SBATCH --output=hpc/logs/%x_%j.out

export MODEL=${MODEL:-qwen3.5:0.8b-bf16}
export SPLIT=dev
export STARTS=${STARTS:-"0 25"} N=${N:-25}
export THINKING=${THINKING:-on}
exec bash hpc/run_episodes.sh
