# run this file to make inference requests: python qdrant/inference.py
# modify QUERY and COLLECTION_NAME as needed
import pprint
from client import get_client, EMBEDDING_MODEL_NAME
from fastembed import TextEmbedding
import os
import datetime

# ------ TODO ------
# modify query to ask what you want
QUERY = "What are the best pair of over the ear headphones for a college student. I am looking for things like long battery life, good noise cancellation, and something under $200-$250. Not too concerned about the audio quality, I need something that will last me a long time and is easy on the go like on the bus, walking around, usually in noisy environments."
COLLECTION_NAME = "headphone_data"
RESULT_COUNT = 3 # top n results to return

client = get_client()

# init embeddings model
print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
print("Model initialized.")

# create vector for the query
print(f"Creating vector for the query: '{QUERY}'")
query_vector = next(embedding_model.embed([QUERY]))

# request qdrant with vector
print("Searching for similar documents in Qdrant...")
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    # query_vector=query_vector.tolist(), 
    limit=RESULT_COUNT,  
    with_payload=True 
)

print("\n--- Top Search Results ---")
out_dir = os.path.join(os.getcwd(), "qdrant", "out")
os.makedirs(out_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(out_dir, f"{timestamp}_qdrant_inference.txt")

with open(out_path, "w") as f:
    for i, result in enumerate(search_results, 1):
        f.write(f"Result {i}:\n")
        f.write(pprint.pformat(result))
        f.write("\n\n")

print(f"Results saved to: {out_path}")