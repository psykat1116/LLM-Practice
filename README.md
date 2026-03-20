# LLM Practice
Basic steps for installing conda environment into the server. Downloading model and datasets from huggingface into the server. Python code for split indexing and find max token size.

### Install conda environment into the server

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### Create environment

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

### Install core packages

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate sentencepiece
pip install huggingface_hub
```

## Download Llama-3.2-3B-Instruct Model

- We can download this using huggingface cli
- ```bash
    conda activate llama
    huggingface-cli login
  ```
- ```bash
    mkdir -p ~/models
    cd ~/models

    huggingface-cli download meta-llama/Llama-3.2-3B \
  --local-dir Llama-3.2-3B \
  --local-dir-use-symlinks False
  ```
- **I faced a issue here I am not able to use huggingface-cli even though I have installed `huggingface_hub`**.
- Therefore, I go through this documentation [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). We also have not generate a new token in huggingface.

  ```bash
  curl -LsSf https://hf.co/cli/install.sh | bash
  hf auth login
  ```
- Also for download the model I use the below code.

  ```python
  from huggingface_hub import login, snapshot_download
  from pathlib import Path

  BASE_DIR = "/home"
  MODEL_DIR = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"

  Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

  #It will automatically done if hf auth login done successfully
  login()

  snapshot_download(
      # We need to get permission in huggingface to download this model
      repo_id="meta-llama/Llama-3.2-3B-Instruct",
      local_dir=MODEL_DIR
  )

  print("Download complete:", MODEL_DIR)
  ```

## Download CNN Dailynews Dataset

- Also, I go through the [cnn-dailymail news data](https://huggingface.co/datasets/abisee/cnn_dailymail) and download this datset using the below python code.
  ```python
  from datasets import load_dataset
  from pathlib import Path

  BASE_DIR = "/home"
  DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"

  Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

  dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
  dataset.save_to_disk(DATA_DIR)

  print("CNN/DailyMail downloaded to:", DATA_DIR)
  ```

## Test GPU Availability

- Additionaly, I additional code to check there is GPU available or not
  ```python
  import torch
  print("CUDA:", torch.cuda.is_available())
  print("GPU:", torch.cuda.get_device_name(0))
  ```

## Split Indexing

- Randomly choose 1000 randomly sampled data points for test and 100 data points for validation from the cnn-dailymail test data points.
  ```python
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
  ```

## Find Max Token Size

- Find max number of token size which is near multiple of 8
  ```python
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
  ```