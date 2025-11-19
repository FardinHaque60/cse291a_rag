# run bi encoder and do initial similarity search for candidate chunks

from lib.embedding_models import bi_encoder_model
from lib.qdrant_client import get_qdrant_client
from lib.constants import COLLECTION_NAME, RESULTS_COUNT

# TODO modify prompt to run unit test
PROCESSED_QUERY = "What does the New York times say is the best wireless headphones?"

# TODO determine what format to give initial rank
def bi_encoder_rank(processed_user_query: str) -> dict:
    '''
        description: Given a processed user query, use a bi-encoder model to find and rank relevant document chunks based on similarity.
            - vectorizes query
            - make inference request to qdrant
            - return formatted qdrant response

        input: processed_user_query, processed from previous preprocessing step (str)
        output: qdrant response in format (TODO)
    '''
    embedding_model = bi_encoder_model()
    client = get_qdrant_client()

    query_vector = next(embedding_model.embed([processed_user_query]))
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=RESULTS_COUNT,   
        with_payload=True 
    )

    # TODO format search_results for cross encoder to use
    
    # returns a list of ScoredPoint objects, example format:
    # ScoredPoint(id='37fcf274-8072-4d8e-bde1-1ee375a4b4d6', version=24, score=0.8005285, payload={'source_file': 'wired_article.pdf', 'page': 2}, vector=None, shard_key=None, order_value=None)

if __name__ == "__main__":
    print(bi_encoder_rank(PROCESSED_QUERY))