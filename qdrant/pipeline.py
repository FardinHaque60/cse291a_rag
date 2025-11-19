import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qdrant.client import EMBEDDING_MODEL_NAME
from fastembed import TextEmbedding
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():
    print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("Model initialized.")
    return embedding_model

def retrieval_pipeline(query, collection_name, results_count, client, reranker_model):
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
    print(f"Search results: {search_results}")

    # rerank with cross-encoder
    print("Reranking results...")
    results = [(query, item.payload["source_file"]) for item in search_results.points]
    scores = [score.item() for score in reranker_model.predict(results)]
    ranked_results = sorted(zip(search_results.points, scores), key=lambda x: x[1], reverse=True)
    return ranked_results
