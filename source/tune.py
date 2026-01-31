import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--max_new_tokens", type=int, required=True)
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--top_p", type=float, required=True)
args = parser.parse_args()

BASE_DIR = "/home"
DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"
MODEL_DIR = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"
INDEX_DIR = f"{BASE_DIR}/indices/validation_indices.json"
RESULT_FILE = f"{BASE_DIR}/outputs/tuning_results.jsonl"

Path(f"{BASE_DIR}/outputs").mkdir(exist_ok=True)
dataset = load_from_disk(DATA_DIR)["test"]

SEED = 42
VAL_SAMPLES = 100
np.random.seed(SEED)
torch.manual_seed(SEED)

with open(INDEX_DIR, "r") as f:
    val_indices = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()

def generate_summary(article):
    messages = [
        {
            "role": "system",
            "content": "You are a summarization engine. Output ONLY the summary text."
        },
        {
            "role": "user",
            "content": (
                "Summarize the following news article in 3–4 sentences.\n\n"
                f"{article}"
            )
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

rouge_sum = 0.0

for idx in val_indices:
    article = dataset[idx]["article"]
    reference = dataset[idx]["highlights"]

    summary = generate_summary(article)
    rouge_sum += scorer.score(reference, summary)["rougeL"].fmeasure

rouge_avg = rouge_sum / VAL_SAMPLES

with open(RESULT_FILE, "a") as f:
    f.write(json.dumps({
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rougeL": rouge_avg
    }) + "\n")

print("ROUGE-L:", rouge_avg)