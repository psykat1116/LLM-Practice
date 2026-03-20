import re
import json
import evaluate
from datasets import load_from_disk

BASE_DIR = "/home"
DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"
PRED_FILES = {
    "icl_random":  f"{BASE_DIR}/outputs/cnn_llama_icl_random.jsonl",
    "icl_similar": f"{BASE_DIR}/outputs/cnn_llama_icl_similar.jsonl",
}

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        records = json.loads(raw)
        return records

    lines = raw.splitlines()
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        buffer += line
        try:
            obj = json.loads(buffer)
            records.append(obj)
            buffer = ""
        except json.JSONDecodeError:
            continue

    if buffer.strip():
        try:
            records.append(json.loads(buffer))
        except json.JSONDecodeError:
            chunks = re.split(r'\}\s*\n\s*\{', raw)
            records = []
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if len(chunks) > 1:
                    if i == 0:
                        chunk = chunk + "}"
                    elif i == len(chunks) - 1:
                        chunk = "{" + chunk
                    else:
                        chunk = "{" + chunk + "}"
                try:
                    records.append(json.loads(chunk))
                except json.JSONDecodeError as e:
                    print(f"  [WARNING] Skipping malformed chunk (chunk {i}): {e}")

    return records


print("Loading dataset ...")
dataset = load_from_disk(DATA_DIR)["test"]

rouge     = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

for strategy, pred_file in PRED_FILES.items():
    references  = []
    predictions = []

    for obj in load_jsonl(pred_file):
        pred = obj["summary"].strip()
        ref  = dataset[obj["id"]]["highlights"].strip()
        if pred:
            predictions.append(pred)
            references.append(ref)

    print(f"Loaded {len(predictions)} samples.\n")
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    
    print("ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"  {k:10s}: {v:.4f}")

    print("\nBERT Scores:")
    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )

    precision = sum(bert_scores["precision"]) / len(bert_scores["precision"])
    recall    = sum(bert_scores["recall"])    / len(bert_scores["recall"])
    f1        = sum(bert_scores["f1"])        / len(bert_scores["f1"])
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")