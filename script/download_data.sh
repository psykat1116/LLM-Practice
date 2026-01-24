#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=success_%j.txt
#SBATCH --error=failure_%j.txt
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=gpupart_l40

source ~/.bashrc
conda activate llama

python data_download.py