#!/bin/bash
# Run a model on the full test split (500 goals, 5 slices of 100).
# Submit from HPC3 (g17): THINKING=off MODEL=qwen3.5-0.8b-sft-e2 sbatch hpc/run_test.sbatch

#SBATCH --job-name=run-test
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:a100:1                    # A100 nodes cs001-cs004; the P100s are too old for CUDA 13
#SBATCH --exclude=cs002                      # its GPUs give CUDA error 999 to new processes (Nov 2026)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G                            # about 14 GB per slice
#SBATCH --time=2-00:00:00
#SBATCH --output=hpc/logs/%x_%j.out

: "${MODEL:?set MODEL}"
export MODEL SPLIT=test STARTS="0 100 200 300 400" N=100
export THINKING=${THINKING:-on}
exec bash hpc/run_episodes.sh
