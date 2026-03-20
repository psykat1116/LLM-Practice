# LLM Week 3
For the purpose of measuring similary I have used tf-idf metric but there is a huge drawback of it. tf-idf cannot capure the context of a word. For that we will MiniLM model to generate the context based embedding then for efficient searching of similarities we are using FAISS IVFFlat Indexing method with uses K-Means for reducing the search space.

## Text Summarization
- Summarize 1000 random data from dataset and summarize them to a 3-4 Sentence summarization.
  ```python
    import json
    import torch
    import faiss
    import random
    import numpy as np
    from pathlib import Path
    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    BASE_DIR   = "/home"
    DATA_DIR   = f"{BASE_DIR}/datasets/cnn_dailymail"
    INDEX_DIR  = f"{BASE_DIR}/indices/test_indices.json"
    MODEL_DIR  = f"{BASE_DIR}/models/Llama-3.2-3B-Instruct"
    OUTPUT_DIR = Path(f"{BASE_DIR}/outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    K_SHOTS       = 3
    SEED          = 42
    ENCODE_BATCH  = 512
    STRATEGIES    = ["random", "similar"]
    EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

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

    EMBED_DEVICE_ID = 1 if torch.cuda.device_count() > 1 else 0
    EMBED_DEVICE    = f"cuda:{EMBED_DEVICE_ID}"

    encoder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    train_embeddings = encoder.encode(
        list(train_articles),
        batch_size = ENCODE_BATCH,
        show_progress_bar = True,
        normalize_embeddings = True,
        convert_to_numpy = True,
    ).astype(np.float32)

    nprobe = 64
    N      = train_embeddings.shape[0]
    DIM    = train_embeddings.shape[1]
    nlist  = min(4096, int(np.sqrt(N)))

    quantizer = faiss.IndexFlatIP(DIM)
    cpu_index = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    cpu_index.train(train_embeddings)
    cpu_index.add(train_embeddings)
    cpu_index.nprobe = nprobe

    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, EMBED_DEVICE_ID, cpu_index)
    del cpu_index, train_embeddings

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()

    def sample_random(test_idx: int, k: int) -> list[int]:
        rng = random.Random(SEED + test_idx)
        pool = list(range(len(train_articles)))
        return rng.sample(pool, k)

    def sample_similar(test_article: str, k: int) -> list[int]:
        query_vec = encoder.encode(
            [test_article],
            normalize_embeddings = True,
            convert_to_numpy = True,
        ).astype(np.float32)

        _, top_k_indices = gpu_index.search(query_vec, k)
        return top_k_indices[0].tolist()

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
                        max_new_tokens=96,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.8,
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

    print("\nAll ICL inference runs complete.")
  ```

## Evaluation Script
- Evaluation of system generated summaries using BERTScore and ROUGE metric.
  ```python
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
  ```

## Evaluation Result
```bash
--------------------------------------------------
  Strategy: icl_random
--------------------------------------------------
ROUGE Scores:
  rouge1    : 0.3962
  rouge2    : 0.1516
  rougeL    : 0.2505
  rougeLsum : 0.3506

BERT Scores:
  Precision: 0.8767
  Recall:    0.8768
  F1:        0.8766

--------------------------------------------------
  Strategy: icl_similar
--------------------------------------------------
ROUGE Scores:
  rouge1    : 0.4007
  rouge2    : 0.1573
  rougeL    : 0.2564
  rougeLsum : 0.3586

BERT Scores:
  Precision: 0.8783
  Recall:    0.8776
  F1:        0.8778
```
