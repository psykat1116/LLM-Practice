from pathlib import Path
from datasets import load_dataset

BASE_DIR = "/home"
DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
dataset.save_to_disk(DATA_DIR)

print("CNN/DailyMail downloaded to:", DATA_DIR)