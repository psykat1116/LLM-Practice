#!/bin/bash
#SBATCH --job-name=llama-tune
#SBATCH --array=0-26
#SBATCH --ntasks=1
#SBATCH --output=logs/output_%j_%a.txt
#SBATCH --error=logs/error_%j_%a.txt
#SBATCH --partition=gpu_l40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00

source ~/.bashrc
conda activate llama

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" params.txt)

MAX_NEW_TOKENS=$(echo $PARAMS | awk '{print $1}')
TEMP=$(echo $PARAMS | awk '{print $2}')
TOP_P=$(echo $PARAMS | awk '{print $3}')

python tune.py \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMP \
  --top_p $TOP_P
