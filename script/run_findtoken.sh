#!/bin/bash
#SBATCH --job-name=find_token
#SBATCH --ntasks=1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=gpu_l40
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1:00:00

source ~/.bashrc
conda activate llama

python findtoken.py
