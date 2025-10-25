import math
import os
import json
import sys
from qdrant_client.http import models
from fastembed import TextEmbedding
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qdrant.client import get_client, EMBEDDING_MODEL_NAME
from datetime import datetime

K = 5 # calculating recall/precision at kth position
L = 10 # limit for how many responses to get from qdrant

def MRR(reranked_lists, ground_truth):
    """
    (CHUNK Level)
    Compute Mean Reciprocal Rank (MRR) for a set of reranked lists.

    Parameters:
    reranked_lists (list of list): A list where each element is a list of reranked item IDs.
    ground_truth (list): A list of ground truth item IDs corresponding to each reranked list.

    Returns:
    float: The Mean Reciprocal Rank score.
    """
    total_reciprocal_rank = 0.0
    num_queries = len(reranked_lists)

    for i, reranked in enumerate(reranked_lists):
        if reranked in ground_truth: 
            return 1.0 / (i + 1)
        
    return 0
    # return total_reciprocal_rank / num_queries if num_queries > 0 else 0.0

def precision_at_k(reranked_lists, ground_truth, k):
    """
    (DOC Level)
    Compute Precision at K for a set of reranked lists.

    Parameters:
    reranked_lists (list of list): A list where each element is a list of reranked item IDs.
    ground_truth (list): A list of ground truth item IDs.
    k (int): The cutoff rank.

    Returns:
    float: The Precision at K score.
    """
    top_k = reranked_lists[:k]
    relevant_items = sum(1 for item in top_k if item in ground_truth)
    return relevant_items / k if k > 0 else 0.0

def recall_at_k(reranked_lists, ground_truth, k):
    """
    (DOC Level)
    Compute Recall at K for a set of reranked lists.

    Parameters:
    reranked_lists (list of list): A list where each element is a list of reranked item IDs.
    ground_truth (list): A list of ground truth item IDs.
    k (int): The cutoff rank.

    Returns:
    float: The Recall at K score.
    """
    top_k = reranked_lists[:k]
    relevant_items = sum(1 for item in top_k if item in ground_truth)
    return relevant_items / len(ground_truth) if len(ground_truth) > 0 else 0.0

def nDCG_at_k(reranked_lists, ground_truth_chunk, qdrant_source_files, ground_truth_doc, k):
    """
    (GRADE Level)
    Compute normalized Discounted Cumulative Gain (nDCG) at K for a set of reranked lists.

    Parameters:
    reranked_lists (list of list): A list where each element is a list of reranked item IDs.
    ground_truth (list): A list of ground truth item IDs.
    k (int): The cutoff rank.

    Returns:
    float: The nDCG at K score.
    """
    def dcg(relevance_scores):
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    relevance_scores = []
    for item, file in zip(reranked_lists[:k], qdrant_source_files[:k]):
        if item in ground_truth_chunk:
            relevance_scores.append(2)  # High relevance for chunk level
        
        elif file in ground_truth_doc:
            relevance_scores.append(1)  # Lower relevance for doc level

        else:
            relevance_scores.append(0)  # No relevance
        
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(ideal_relevance_scores)

    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return ndcg

def read_data_from_file(file_path):
    """
    Utility function to read data from a file.

    Parameters:
    file_path (str): Path to the file.

    Returns:
    list: List of lines read from the file.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [int(line.strip()) for line in data]

def get_metric_from_relevance(qdrant_results, ground_truths, qdrant_files, qdrant_gold_files, latency): 
    mrr = MRR(qdrant_results, ground_truths)
    precision = precision_at_k(qdrant_results, ground_truths,  K)
    recall = recall_at_k( qdrant_results, ground_truths, K)
    # grade relevance = doc + chunk
    ndcg = nDCG_at_k(qdrant_results, ground_truths, qdrant_files, qdrant_gold_files, K)
    return {
        "Reciprocal Rank": mrr,
        "Precision@K": precision,
        "Recall@K": recall,
        "nDCG@K": ndcg,
        "Latency (milliseconds)": round(latency*1000, 3)
    }

def evaluate_metrics(doc_relevance_file, chunk_relevance_file, qdrant_results_file):
    doc_relevance = read_data_from_file(doc_relevance_file)
    chunk_relevance = read_data_from_file(chunk_relevance_file)
    qdrant_results = read_data_from_file(qdrant_results_file)

    metrics = get_metric_from_relevance(doc_relevance, chunk_relevance, qdrant_results)
    return metrics

if __name__ == "__main__":
    '''
    doc_relevance_file = "example/doc_relevances.txt"
    chunk_relevance_file = "example/chunk_relevances.txt"
    qdrant_results_file = "example/qdrant_results.txt"

    metrics = evaluate_metrics(doc_relevance_file, chunk_relevance_file, qdrant_results_file)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}") 
    '''

    client = get_client()
    print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("Model initialized.")

    json_path = os.path.join(os.getcwd(), "metric", "prompts.json")
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
            query_vector = next(embedding_model.embed([prompt]))

            start = time.time()
            search_results = client.query_points(
                collection_name="production_data",
                query=query_vector.tolist(),
                limit=L,  
                with_payload=True 
            )
            end = time.time()

            qdrant_ids = []
            qdrant_files = []
            for point in search_results:
                point_list = point[1]
                for p in point_list:
                    qdrant_ids.append(p.id)
                    qdrant_files.append(p.payload["source_file"]) # check if field exists

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
                "qdrant_ids": qdrant_ids,
                "gold_set": gold_ids,
                "qdrant_files": qdrant_files,
                "gold_files": gold_file,
                "metrics": metrics
            }

            results.append(entry)

    for metric in avg_metrics:
        avg_metrics[metric] = round(avg_metrics[metric]/prompts, 3)
    results.append(avg_metrics)

    # prepare output directory and persistent file path for the run
    out_dir = os.path.join(os.getcwd(), "metric", "out")
    os.makedirs(out_dir, exist_ok=True)
    if 'metrics_file_path' not in globals():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file_path = os.path.join(out_dir, f"{timestamp}_metrics.json")

    with open(metrics_file_path, "a", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
        out_f.write("\n\n")
    
    print("metrics wrote to out/ directory")