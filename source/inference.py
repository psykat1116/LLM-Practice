import json
import torch
import random
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = "/home"
DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"
INDEX_DIR = f"{BASE_DIR}/indices/test_indices.json"
MODEL_DIR = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"
OUTPUT_FILE = f"{BASE_DIR}/outputs/cnn_llama_inference.jsonl"
Path(f"{BASE_DIR}/outputs").mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

dataset = load_from_disk(DATA_DIR)["test"]
with open(INDEX_DIR, "r") as f:
    indices = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype = torch.float16,
    device_map = "auto",
)
model.eval()

with open(OUTPUT_FILE, "w") as f:
    for idx in indices:
        article = dataset[idx]["article"]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarization engine. "
                    "Output ONLY the summary text. "
                    "Do not add introductions, role names, or explanations."
                )
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following news article in 3–4 sentences.\n\n"
                    f"{article}"
                )
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors = "pt"
        ).to(model.device)

        # Keep the best hyper parameter
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens = 96,
                do_sample = True,
                temperature = 0.3,
                top_p = 0.8
            )

        gen_tokens = outputs[0][inputs.shape[-1]:]
        summary = tokenizer.decode(gen_tokens, skip_special_tokens = True).strip()

        f.write(json.dumps({
            "id": idx,
            "summary": summary
        }) + "\n")

print("Inference complete.")
print("Saved to:", OUTPUT_FILE)
