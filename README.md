# LLM Week 1

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

## Inference Script
- Randomly choose 1000 randomly sampled data points from the cnn-dailymail test data points and run through the LLM using below code
    ```python
    import json
    import torch
    import random
    from pathlib import Path
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM

    BASE_DIR = "/home"
    MODEL_DIR = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"
    DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"
    OUTPUT_FILE = f"{BASE_DIR}/outputs/cnn_llama_inference.jsonl"

    MAX_NEW_TOKENS = 128
    NUMBER_SAMPLES = 1000

    dataset = load_from_disk(DATA_DIR)["test"]
    indices = random.sample(range(len(dataset)), NUMBER_SAMPLES)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
    	MODEL_DIR,
    	dtype=torch.float16,
    	device_map="auto"
    )

    model.eval()

    Path(f"{BASE_DIR}/outputs").mkdir(exist_ok = True)

    with open(OUTPUT_FILE, "w") as f:
    	for index in indices:
    		article = dataset[index]["article"]
    		prompt = (
     		   "You are a helpful assistant.\n\n"
       		   "Summarize the following news article in 3–4 sentences.\n\n"
       		   f"Article:\n{article}\n\nSummary:\n"
    		)

    		inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    		with torch.no_grad():
    			outputs = model.generate(
    				**inputs,
    				max_new_tokens = MAX_NEW_TOKENS,
    				do_sample = True,
    				temperature = 0.7,
    				top_p = 0.9
    			)
    		generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    		summary = tokenizer.decode(generated_tokens, skip_special_tokens = True).strip()

    		f.write(json.dumps({
    			"id": index,
    			"article": article,
    			"summary": summary
    		}) + "\n")

    print("Inference Complete. Saved To: ", OUTPUT_FILE)
    ```

## Evaluation Script
- Evaluation of system generated summaries using BERTScore and ROUGE metric.
    ```python
    import json
    import evaluate
    from datasets import load_from_disk

    BASE_DIR = "/home"
    DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"
    PRED_FILE = f"{BASE_DIR}/outputs/cnn_llama_inference.jsonl"

    references = []
    predictions = []
    dataset = load_from_disk(DATA_DIR)["test"]

    with open(PRED_FILE, "r") as f:
    	for line in f:
    		obj = json.loads(line)
    		predictions.append(obj["summary"])
    		references.append(dataset[obj["id"]]["highlights"])

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions = predictions, references = references)

    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(
    	predictions = predictions,
    	references = references,
    	lang = "en"
    )

    print("ROUGE Scores:")
    for k, v in rouge_scores.items():
    	print(f"{k}: {v:.4f}")

    print("\nBERTScore:")
    print(f"Precision: {sum(bert_scores['precision'])/len(bert_scores['precision']):.4f}")
    print(f"Recall:    {sum(bert_scores['recall'])/len(bert_scores['recall']):.4f}")
    print(f"F1:        {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
    ```

## Evaluation Result
```bash
    ROUGE Scores:
    rouge1: 0.3537
    rouge2: 0.1299
    rougeL: 0.2211
    rougeLsum: 0.2864

    BERT Scores:
    Precision: 0.8649
    Recall:    0.8775
    F1:        0.8711
```