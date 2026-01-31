import json
import random
from pathlib import Path
from datasets import load_from_disk

BASE_DIR = "/home"
OUTPUT_DIR = f"{BASE_DIR}/indices"
DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"

SEED = 42
NUM_TEST = 1000
NUM_VALIDATION = 100

random.seed(SEED)
dataset = load_from_disk(DATA_DIR)
test_dataset = dataset["test"]

total_indices = list(range(len(test_dataset)))
random.shuffle(total_indices)

validation_indices = total_indices[:NUM_VALIDATION]
test_indices = total_indices[NUM_VALIDATION:NUM_VALIDATION + NUM_TEST]

Path(OUTPUT_DIR).mkdir(exist_ok=True)

with open(f"{OUTPUT_DIR}/validation_indices.json", "w") as f:
    json.dump(validation_indices, f)

with open(f"{OUTPUT_DIR}/test_indices.json", "w") as f:
    json.dump(test_indices, f)

print(f"Validation indices: {len(validation_indices)} samples")
print(f"Test indices: {len(test_indices)} samples")
print(f"Overlap check: {len(set(validation_indices) & set(test_indices))} (should be 0)")
print(f"Indices saved to {OUTPUT_DIR}")