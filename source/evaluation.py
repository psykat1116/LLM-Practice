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
