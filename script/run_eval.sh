#!/bin/bash
#SBATCH --job-name=llama-infer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=gpupart_l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00

source ~/.bashrc
conda activate llama

python evaluation.py