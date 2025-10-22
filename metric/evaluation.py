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

def nDCG_at_k(reranked_lists, ground_truth_chunk, ground_truth_doc, k):
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
    for item in reranked_lists[:k]:
        if item in ground_truth_chunk:
            relevance_scores.append(2)  # High relevance for chunk level
        elif item in ground_truth_doc:
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

def get_metric_from_relevance(doc_relevance, chunk_relevance, qdrant_results): 
    K = 10
    mrr = MRR(qdrant_results, chunk_relevance)
    precision = precision_at_k(qdrant_results,doc_relevance,  K)
    recall = recall_at_k( qdrant_results,doc_relevance, K)
    # grade relevance = doc + chunk
    ndcg = nDCG_at_k(qdrant_results, chunk_relevance, doc_relevance, K)
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
    doc_relevance_file = "example/doc_relevances.txt"
    chunk_relevance_file = "example/chunk_relevances.txt"
    qdrant_results_file = "example/qdrant_results.txt"

    metrics = evaluate_metrics(doc_relevance_file, chunk_relevance_file, qdrant_results_file)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}") 