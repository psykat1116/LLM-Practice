import json
import torch
import random
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR   = "/home"
OUTPUT_DIR = Path(f"{BASE_DIR}/outputs")
DATA_DIR   = f"{BASE_DIR}/datasets/cnn_dailymail"
INDEX_DIR  = f"{BASE_DIR}/indices/test_indices.json"
MODEL_DIR  = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"
OUTPUT_DIR.mkdir(exist_ok=True)

K_SHOTS       = 3
SEED          = 42
STRATEGIES    = ["random", "similar"]

random.seed(SEED)
np.random.seed(SEED)

print("Loading dataset …")
full_dataset = load_from_disk(DATA_DIR)
test_dataset  = full_dataset["test"]
train_dataset = full_dataset["train"]

with open(INDEX_DIR, "r") as f:
    test_indices = json.load(f)

train_articles   = train_dataset["article"]
train_highlights = train_dataset["highlights"]

vectorizer    = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
train_tfidf   = vectorizer.fit_transform(train_articles)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

def sample_random(test_idx: int, k: int) -> list[int]:
    rng = random.Random(SEED + test_idx)
    pool = list(range(len(train_articles)))
    return rng.sample(pool, k)

def sample_similar(test_article: str, k: int) -> list[int]:
    test_vec = vectorizer.transform([test_article])
    sims     = cosine_similarity(test_vec, train_tfidf)[0]
    top_k    = np.argsort(sims)[::-1][:k]
    return top_k.tolist()

SYSTEM_PROMPT = (
    "You are a summarization engine. "
    "Output ONLY the summary text. "
    "Do not add introductions, role names, or explanations."
)

def build_messages(test_article: str, demo_indices: list[int]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for di in demo_indices:
        demo_article  = train_articles[di]
        demo_summary  = train_highlights[di]
        messages.append({
            "role": "user",
            "content": (
                "Summarize the following news article in 3–4 sentences.\n\n"
                f"{demo_article}"
            )
        })
        messages.append({
            "role": "assistant",
            "content": demo_summary
        })

    messages.append({
        "role": "user",
        "content": (
            "Summarize the following news article in 3–4 sentences.\n\n"
            f"{test_article}"
        )
    })

    return messages

for strategy in STRATEGIES:
    output_file = OUTPUT_DIR/f"cnn_llama_icl_{strategy}.jsonl"
    sampler     = sample_random if strategy == "random" else sample_similar

    print(f"\nRunning {strategy.upper()} {K_SHOTS}-shot ICL → {output_file}")
    with open(output_file, "w") as f:
        for step, idx in enumerate(test_indices, 1):
            article = test_dataset[idx]["article"]

            if strategy == "random":
                demo_idxs = sampler(idx, K_SHOTS)
            else:
                demo_idxs = sampler(article, K_SHOTS)

            messages = build_messages(article, demo_idxs)
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    top_p = 0.8,
                    do_sample = True,
                    temperature = 0.3,
                    max_new_tokens = 96,
                )

            gen_tokens = outputs[0][inputs.shape[-1]:]
            summary    = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            record = {
                "id":           idx,
                "strategy":     strategy,
                "demo_indices": demo_idxs,
                "summary":      summary,
            }
            f.write(json.dumps(record, indent=4) + "\n")

            if step % 50 == 0:
                print(f"  [{strategy}] {step}/{len(test_indices)} done")

    print(f"  Saved to {output_file}")