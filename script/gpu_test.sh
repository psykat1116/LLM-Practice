#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --partition=gpupart_l40
#SBATCH --time=00:02:00
#SBATCH --output=gpu_test.txt
#SBATCH --error=gpu_error.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

source ~/.bashrc
conda activate llama

python - <<EOF
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF
