# cross encoder rerank for initial candidate chunks

from .p2_bi_encoder_rank import bi_encoder_rank
from lib.embedding_models import cross_encoder_model

# TODO modify prompt to run unit test
PROCESSED_QUERY = ""

''' try cross encoder with multiple options:
- summary
- keywords
- raw text
'''

# TODO determine format for initial chunks
def cross_encoder_rerank(initial_chunks: dict, processed_query: str) -> dict:
    '''
        description: Given initial candidate chunks from bi-encoder, use a cross-encoder model to rerank them for better relevance.
            - use cross encoder to embed query and chunk summaries
            - use cross encoder scores to rerank initial chunks

        input: initial_chunks, candidate chunks from bi-encoder step (TODO)
        output: reranked chunks in format (TODO)
    '''
    cross_encoder = cross_encoder_model()

    initial_chunk_summaries = None # TODO parse initial_chunks to get only summaries
    score = cross_encoder.rerank(processed_query, initial_chunk_summaries)

    for i in score:
        print(i)

if __name__ == "__main__":
    # call bi_encoder_rank to get initial candidate chunks
    # can modify to pass in hard coded initial chunks if needed
    initial_chunks = bi_encoder_rank(PROCESSED_QUERY)
    rerank_result = cross_encoder_rerank(initial_chunks, PROCESSED_QUERY)

    print(rerank_result)