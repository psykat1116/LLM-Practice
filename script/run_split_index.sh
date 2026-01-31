#!/bin/bash
#SBATCH --job-name=split_index
#SBATCH --ntasks=1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=gpupart_l40
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate llama

python split_index.py
