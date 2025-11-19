# run this file to make inference requests: python qdrant/inference.py
# modify QUERY and COLLECTION_NAME as needed
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pprint
from qdrant.client import get_client, get_reranker
import os
import datetime
import time
# import retrieval pipeline as absolute path using importlib
from qdrant.pipeline import retrieval_pipeline

# ------ TODO ------
# modify query to ask what you want
QUERY = "What does the New York times say is the best wireless headphones?"
COLLECTION_NAME = "production_data"
RESULT_COUNT = 10 # top n results to return

client = get_client()
reranker_model = get_reranker()
# request qdrant with vector
start = time.time()
print("Searching for similar documents in Qdrant...")
ranked_results = retrieval_pipeline(QUERY, COLLECTION_NAME, RESULT_COUNT, client, reranker_model)
end = time.time()
print(f"Search completed in {end - start:.2f} seconds.")

print("\n--- Top Search Results ---")
out_dir = os.path.join(os.getcwd(), "qdrant", "out")
os.makedirs(out_dir, exist_ok=True)
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