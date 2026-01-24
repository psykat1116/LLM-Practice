#!/bin/bash
#SBATCH --job-name=downlaod_llama
#SBATCH --output=success_%j.txt
#SBATCH --error=failure_%j.txt
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=gpupart_l40

source ~/.bashrc
conda activate llama

python model_download.py