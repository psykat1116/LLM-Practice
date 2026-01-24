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
