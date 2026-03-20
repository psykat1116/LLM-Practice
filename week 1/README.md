# LLM Week 1

## Hyperparameter Tuning

- Take all 27 combinations of max_token_size, temperature, top_p find the best based upon RougeL Score metrics
  ```python
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
  ```

- Script For Running 27 jobs in parallel in Server
  ```bash
    #!/bin/bash
    #SBATCH --job-name=llama-tune
    #SBATCH --array=0-26
    #SBATCH --ntasks=1
    #SBATCH --output=logs/output_%j_%a.txt
    #SBATCH --error=logs/error_%j_%a.txt
    #SBATCH --partition=gpu_l40
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=8:00:00

    source ~/.bashrc
    conda activate llama

    PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" params.txt)

    MAX_NEW_TOKENS=$(echo $PARAMS | awk '{print $1}')
    TEMP=$(echo $PARAMS | awk '{print $2}')
    TOP_P=$(echo $PARAMS | awk '{print $3}')

    python tune.py \
      --max_new_tokens $MAX_NEW_TOKENS \
      --temperature $TEMP \
      --top_p $TOP_P
  ```

## Text Summarization

- Summarize 1000 random data from dataset and summarize them to a 3-4 Sentence summarization.
  ```python
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
rouge1: 0.3795
rouge2: 0.1449
rougeL: 0.2391
rougeLsum: 0.3077

BERTScore:
Precision: 0.8699
Recall:    0.8731
F1:        0.8714
```
