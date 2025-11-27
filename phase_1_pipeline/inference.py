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
from tqdm import tqdm

# modify query to ask what you want
QUERIES = [
    "What does the New York times say is the best wireless headphones?",
    "I'm thinking of getting an ASUS ROG gaming laptop, how much would it cost?"
]
COLLECTION_NAME = "production_data"
RESULT_COUNT = 10 # top n results to return

@lru_cache(maxsize=1)
def get_embedding_model():
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return embedding_model

def retrieval_pipeline(query, collection_name, results_count, client):
    embedding_model = get_embedding_model()

    # create vector for the query
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
    out_dir = os.path.join(os.getcwd(), "phase_1_pipeline", "results")
    os.makedirs(out_dir, exist_ok=True)  # Create directory if it does not exist
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{timestamp}_phase1_results.txt")

    with open(out_path, "w") as f:
        for query in tqdm(QUERIES, desc="processing queries"):
            # request qdrant with vectorized query
            start = time.time()
            ranked_results = retrieval_pipeline(query, COLLECTION_NAME, RESULT_COUNT, client)
            end = time.time()

            f.write(f"Query: {query}\n")
            f.write("chunks returned:\n")
            for result in ranked_results.points:
                f.write(pprint.pformat(result))
                f.write("\n")
            f.write(f"\nLatency: {end - start:.2f} seconds\n")
            f.write("-" * 40 + "\n")
                
        print(f"Results saved to: {out_path}")