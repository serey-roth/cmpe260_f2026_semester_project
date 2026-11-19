#!/bin/bash
# Dev baseline of a model on the first 50 dev goals (2 slices of 25), the same goals as our other baselines.
# The Ollama server and the slices are run by hpc/run_episodes.sh.
# The model must already be pulled into ~/ollama-models (see hpc/README.md, step 5).
# Submit from HPC3 (g17): MODEL=qwen3.5:0.8b-bf16 sbatch hpc/run_baseline.sbatch

#SBATCH --job-name=run-baseline-dev
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=g[1,3-5,9,11-12,14-16]   # P100 nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                            # about 14 GB per slice
#SBATCH --time=4:00:00
#SBATCH --output=hpc/logs/%x_%j.out

export MODEL=${MODEL:-qwen3.5:0.8b-bf16}
export SPLIT=dev STARTS="0 25" N=25
export THINKING=${THINKING:-on}
exec bash hpc/run_episodes.sh
