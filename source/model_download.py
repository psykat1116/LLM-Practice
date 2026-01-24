from pathlib import Path
from huggingface_hub import login, snapshot_download

BASE_DIR = "/home"
MODEL_DIR = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

login()

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir=MODEL_DIR
)

print("Download complete:", MODEL_DIR)