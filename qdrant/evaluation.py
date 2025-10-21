import math
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
            total_reciprocal_rank += 1.0 / (i + 1)

    return total_reciprocal_rank / num_queries if num_queries > 0 else 0.0

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
    total_precision = 0.0
    num_queries = len(reranked_lists)
    top_k = reranked_lists[:k]
    relevant_items = sum(1 for item in top_k if item in ground_truth)
    total_precision += relevant_items / k

    return total_precision / num_queries if num_queries > 0 else 0.0

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
    total_recall = 0.0
    num_queries = len(reranked_lists)
    top_k = reranked_lists[:k]
    relevant_items = sum(1 for item in top_k if item in ground_truth)
    total_recall += relevant_items / len(ground_truth) if len(ground_truth) > 0 else 0.0

    return total_recall / num_queries if num_queries > 0 else 0.0

def nDCG_at_k(reranked_lists, ground_truth, k):
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

    total_ndcg = 0.0
    num_queries = len(reranked_lists)

    relevance_scores = [1 if item in ground_truth else 0 for item in reranked_lists[:k]]
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(ideal_relevance_scores)

    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    total_ndcg += ndcg

    return total_ndcg / num_queries if num_queries > 0 else 0.0

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

def get_metric_from_relevance(doc_relevance, chunk_relevance, qdrant_results): 
    K = 10
    mrr = MRR(chunk_relevance, qdrant_results)
    precision = precision_at_k(doc_relevance, qdrant_results, K)
    recall = recall_at_k(doc_relevance, qdrant_results, K)
    # grade relevance = doc + chunk
    grade_relevance = [doc + chunk for doc, chunk in zip(doc_relevance, chunk_relevance)]
    ndcg = nDCG_at_k(grade_relevance, qdrant_results, K)
    return {
        "MRR": mrr,
        "Precision@K": precision,
        "Recall@K": recall,
        "nDCG@K": ndcg
    }

def evaluate_metrics(doc_relevance_file, chunk_relevance_file, qdrant_results_file):
    doc_relevance = read_data_from_file(doc_relevance_file)
    chunk_relevance = read_data_from_file(chunk_relevance_file)
    qdrant_results = read_data_from_file(qdrant_results_file)

    metrics = get_metric_from_relevance(doc_relevance, chunk_relevance, qdrant_results)
    return metrics

if __name__ == "__main__":
    doc_relevance_file = "doc_relevances.txt"
    chunk_relevance_file = "chunk_relevances.txt"
    qdrant_results_file = "qdrant_results.txt"

    metrics = evaluate_metrics(doc_relevance_file, chunk_relevance_file, qdrant_results_file)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}") 