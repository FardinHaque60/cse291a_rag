# run bi encoder and do initial similarity search for candidate chunks
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_2_pipeline.lib.embedding_models import bi_encoder_model
from phase_2_pipeline.lib.qdrant_client import get_qdrant_client
from phase_2_pipeline.lib.constants import RESULTS_COUNT

# TODO modify prompt to run unit test
PROCESSED_QUERY = {
    "query": "What does the New York times say is the best wireless headphones?",
    "collection": "production_data"
}

# TODO determine what format to give initial rank
def bi_encoder_rank(processed_user_query: dict) -> dict:
    '''
        description: Given a processed user query, use a bi-encoder model to find and rank relevant document chunks based on similarity.
            - vectorizes query
            - make inference request to qdrant
            - return formatted qdrant response

        input: {"query": processed from previous preprocessing step, "collection": what collection to take data from} (dict)
        output: qdrant response in format (TODO)
    '''
    embedding_model = bi_encoder_model()
    client = get_qdrant_client()

    query_vector = next(embedding_model.embed([processed_user_query['query']]))
    search_results = client.query_points(
        collection_name=processed_user_query['collection'],
        query=query_vector.tolist(),
        limit=RESULTS_COUNT,   
        with_payload=True 
    )

    return [point for point in search_results.points]

if __name__ == "__main__":
    # print(bi_encoder_rank(PROCESSED_QUERY))
    items = bi_encoder_rank(PROCESSED_QUERY)
    # source_files = [point for point in items.points] # payload.get('source_file') 
    print(items)