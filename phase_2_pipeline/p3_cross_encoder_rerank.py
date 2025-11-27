# cross encoder rerank for initial candidate chunks
import os
import sys
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_2_pipeline.p2_bi_encoder_rank import bi_encoder_rank
from phase_2_pipeline.lib.embedding_models import cross_encoder_model
from phase_2_pipeline.lib.constants import FINAL_COUNT

# TODO modify prompt to run unit test
PROCESSED_QUERY = {
    "query": "What does the New York times say is the best wireless headphones?",
    "collection": "production_data"
}

''' try cross encoder with multiple options:
- summary
- keywords
- raw text
'''

def cross_encoder_rerank(initial_chunks: list, processed_query: str) -> dict:
    '''
        description: Given initial candidate chunks from bi-encoder, use a cross-encoder model to rerank them for better relevance.
            - use cross encoder to embed query and chunk summaries
            - use cross encoder scores to rerank initial chunks

        input: initial_chunks, candidate chunks from bi-encoder step as a list of ScorePoint objs
        output: reranked chunks in 
    '''
    cross_encoder = cross_encoder_model()

    initial_chunk_summaries = []
    for item in initial_chunks:
        # qdrant_ids.append(point.id)
        initial_chunk_summaries.append(item.payload.get("text"))  # Use empty string if "summary" not present

    score = cross_encoder.rerank(processed_query['query'], initial_chunk_summaries)
    scores = [i for i in score]

    scored_summaries = list(zip(initial_chunks, scores))
    scored_summaries.sort(key=lambda x: x[1], reverse=True)
    # Return the sorted summaries and their scores
    return scored_summaries[:FINAL_COUNT]

if __name__ == "__main__":
    # call bi_encoder_rank to get initial candidate chunks
    # can modify to pass in hard coded initial chunks if needed
    initial_chunks = bi_encoder_rank(PROCESSED_QUERY)
    rerank_result = cross_encoder_rerank(initial_chunks, PROCESSED_QUERY)

    pprint(rerank_result)