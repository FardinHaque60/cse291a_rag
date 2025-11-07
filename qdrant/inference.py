# run this file to make inference requests: python qdrant/inference.py
# modify QUERY and COLLECTION_NAME as needed
import pprint
from client import get_client, EMBEDDING_MODEL_NAME
from fastembed import TextEmbedding
import os
import datetime
import time
from sentence_transformers import CrossEncoder


# ------ TODO ------
# modify query to ask what you want
QUERY = "What does the New York times say is the best wireless headphones?"
COLLECTION_NAME = "production_data"
RESULT_COUNT = 10 # top n results to return

client = get_client()

# init embeddings model
print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
print("Model initialized.")

# create vector for the query
print(f"Creating vector for the query: '{QUERY}'")
query_vector = next(embedding_model.embed([QUERY]))

# request qdrant with vector
start = time.time()
print("Searching for similar documents in Qdrant...")
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    # query_vector=query_vector.tolist(), 
    limit=RESULT_COUNT,  
    with_payload=True 
)
end = time.time()
# rerank with cross-encoder
reranker_model = CrossEncoder("BAAI/bge-reranker-base")
print("Reranking results...")
results = [(QUERY, item.payload["source_file"]) for item in search_results.points]
scores = [score.item() for score in reranker_model.predict(results)]
ranked_results = sorted(zip(search_results.points, scores), key=lambda x: x[1], reverse=True)

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