# run this file to make inference requests: python qdrant/inference.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase_1_pipeline.client import get_client
import os
import pprint
import datetime
import time
from phase_1_pipeline.client import EMBEDDING_MODEL_NAME
from fastembed import TextEmbedding
from functools import lru_cache

# modify query to ask what you want
QUERY = "What does the New York times say is the best wireless headphones?"
COLLECTION_NAME = "production_data"
RESULT_COUNT = 10 # top n results to return

@lru_cache(maxsize=1)
def get_embedding_model():
    print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("Model initialized.")
    return embedding_model

def retrieval_pipeline(query, collection_name, results_count, client):
    embedding_model = get_embedding_model()

    # create vector for the query
    print(f"Creating vector for the query: '{query}'")
    query_vector = next(embedding_model.embed([query]))
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        # query_vector=query_vector.tolist(), 
        limit=results_count,  
        with_payload=True 
    )

    return search_results

if __name__ == "__main__":
    client = get_client()
    # reranker_model = get_reranker()

    # request qdrant with vectorized query
    start = time.time()
    ranked_results = retrieval_pipeline(QUERY, COLLECTION_NAME, RESULT_COUNT, client)
    end = time.time()
    print(f"Search completed in {end - start:.2f} seconds.")

    print("\n--- Saving Output to File ---")
    out_dir = os.path.join(os.getcwd(), "phase_1_pipeline", "out")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{timestamp}_qdrant_inference.txt")
    latency_path = os.path.join(out_dir, f"{timestamp}_qdrant_latency.txt")

    with open(out_path, "a") as f:
        for i, result in enumerate(ranked_results):
            f.write(f"Result {i}:\n")
            f.write(pprint.pformat(result))
            f.write("\n\n")

    with open(latency_path, "w") as f:
        f.write(f"Query: {QUERY}\n")
        f.write(f"Latency: {end - start:.2f} seconds\n")
        
    print(f"Results saved to: {out_path}")