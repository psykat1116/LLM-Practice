import random
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

BASE_DIR = "/home"
DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"
MODEL_DIR = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"

SEED = 42
VAL_SAMPLES = 100
random.seed(SEED)

dataset = load_from_disk(DATA_DIR)["test"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
val_indices = random.sample(range(len(dataset)), VAL_SAMPLES)

def round_to_multiple_of_8(x):
    return int(round(x / 8) * 8)

lengths = []
for idx in val_indices:
    ref = dataset[idx]["highlights"]
    token_len = len(tokenizer(ref)["input_ids"])
    lengths.append(token_len)

p50 = round_to_multiple_of_8(np.percentile(lengths, 50))
p75 = round_to_multiple_of_8(np.percentile(lengths, 75))
p90 = round_to_multiple_of_8(np.percentile(lengths, 90))

print("MAX_NEW_TOKENS candidates (rounded to /8):")
print("50th percentile:", p50)
print("75th percentile:", p75)
print("90th percentile:", p90)
