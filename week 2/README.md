# LLM Week 2
Instead of taking plain inference we now using in-context learing for better result. I am using two types of inference one is randomly choosing 3 samples and another way is choosing based on higher similalarity.

## Text Summarization

- Summarize 1000 random data from dataset and summarize them to a 3-4 Sentence summarization.
  ```python
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
  ```

## Evaluation Script

- Evaluation of system generated summaries using BERTScore and ROUGE metric.
  ```python
  import json
    import evaluate
    from datasets import load_from_disk

    BASE_DIR = "/home"
    DATA_DIR = f"{BASE_DIR}/datasets/cnn_dailymail"

    PRED_FILES = {
        "icl_random":  f"{BASE_DIR}/outputs/cnn_llama_icl_random.jsonl",
        "icl_similar": f"{BASE_DIR}/outputs/cnn_llama_icl_similar.jsonl",
    }

    print("Loading dataset ...")
    dataset = load_from_disk(DATA_DIR)["test"]

    rouge     = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    for strategy, pred_file in PRED_FILES.items():
        print(f"\n{'=' * 50}")
        print(f"  Strategy: {strategy}")
        print(f"{'=' * 50}")

        references  = []
        predictions = []
        with open(pred_file, "r") as f:
            for line in f:
                obj  = json.loads(line)
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
        recall    = sum(bert_scores["recall"]) / len(bert_scores["recall"])
        f1        = sum(bert_scores["f1"]) / len(bert_scores["f1"])
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")

    print(f"\n{'=' * 50}")
    print("Evaluation complete.")
  ```

## Evaluation Result

```bash
==================================================
  Strategy: icl_random
==================================================
Loaded 1000 samples.

ROUGE Scores:
  rouge1    : 0.3943
  rouge2    : 0.1497
  rougeL    : 0.2491
  rougeLsum : 0.3502

BERT Scores:
  Precision: 0.8767
  Recall:    0.8764
  F1:        0.8764

==================================================
  Strategy: icl_similar
==================================================
Loaded 1000 samples.

ROUGE Scores:
  rouge1    : 0.4013
  rouge2    : 0.1557
  rougeL    : 0.2562
  rougeLsum : 0.3602

BERT Scores:
  Precision: 0.8786
  Recall:    0.8774
  F1:        0.8779

==================================================
```
