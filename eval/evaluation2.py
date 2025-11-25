## automated evals for phase 2
import os
import json
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase_2_pipeline.lib.qdrant_client import get_qdrant_client
from phase_2_pipeline.lib.embedding_models import bi_encoder_model
from eval.metric_lib import get_metric_from_relevance, L
from phase_2_pipeline.p0_runner import run_pipeline

if __name__ == "__main__":
    client = get_qdrant_client()
    embedding_model = bi_encoder_model()
    print("Embedding Model initialized.")

    json_path = os.path.join(os.getcwd(), "eval", "prompts.json")
    evals_file = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            evals_file = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")

    results = []
    prompts = 0

    avg_metrics = {
        "Reciprocal Rank": 0,
        "Precision@K": 0,
        "Recall@K": 0,
        "nDCG@K": 0,
        "Latency (milliseconds)": 0
    }
    for category in evals_file:
        for prompt, gold_ids, gold_file in zip(category["prompts"], category["gold_set"], category["gold_files"]):
            prompts += 1

            start = time.time()
            raw_results = run_pipeline(prompt)
            search_results = raw_results[1]
            end = time.time()

            qdrant_ids = []
            qdrant_files = []
            for item in search_results:
                qdrant_ids.append(item.id)
                qdrant_files.append(item.payload["source_file"]) # check if field exists

            if not isinstance(gold_ids, list):
                gold_ids = [gold_ids]

            # compute metrics
            latency = end - start
            metrics = get_metric_from_relevance(qdrant_ids, gold_ids, qdrant_files, gold_file, latency)
            for metric in metrics:
                avg_metrics[metric] += metrics[metric]

            # prepare content to append
            entry = {
                "prompt": prompt,
                "llm_response": raw_results[0],
                "qdrant_ids": qdrant_ids,
                "gold_set": gold_ids,
                "qdrant_files": qdrant_files,
                "gold_files": gold_file,
                "metrics": metrics,
            }

            results.append(entry)
    print(f"Evaluated {prompts} prompts.")
    for metric in avg_metrics:
        avg_metrics[metric] = round(avg_metrics[metric]/prompts, 3)
    results.append(avg_metrics)

    # prepare output directory and persistent file path for the run
    out_dir = os.path.join(os.getcwd(), "eval", "out")
    os.makedirs(out_dir, exist_ok=True)
    if 'metrics_file_path' not in globals():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file_path = os.path.join(out_dir, f"{timestamp}_metrics_phase2.json")

    with open(metrics_file_path, "a", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
        out_f.write("\n\n")
    
    print("metrics wrote to out/ directory")