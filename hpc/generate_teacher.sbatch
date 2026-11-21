#!/bin/bash
# Generate teacher episodes for SFT (Qwen3.5-9B with thinking on) against train goals 0–599 in 6 slices of 100.
# Submit from HPC3 (g17): sbatch hpc/generate_teacher.sbatch

#SBATCH --job-name=generate-teacher-data
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:a100:1                    # A100 nodes cs001-cs004; the P100s are too old for CUDA 13
#SBATCH --exclude=cs002                      # its GPUs give CUDA error 999 to new processes (Nov 2026)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                           # each slice loads the full catalog into RAM
#SBATCH --time=7-00:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --mail-user=serey.roth@sjsu.edu
#SBATCH --mail-type=END,FAIL

export MODEL=${MODEL:-qwen3.5:9b}
export SPLIT=${SPLIT:-train}
export STARTS=${STARTS:-"0 100 200 300 400 500"}   # 600 of the 10,538 train goals
export N=${N:-100}                                  # goals per slice
export THINKING=${THINKING:-on}
exec bash hpc/run_episodes.sh
