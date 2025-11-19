# final output generation given final chunks
from .p2_bi_encoder_rank import bi_encoder_rank
from .p3_cross_encoder_rerank import cross_encoder_rerank

# TODO modify processed prompt passed into bi-encoder -> cross encoder, so chunk ranks can be used
PROCESSED_QUERY = ""

def output_generation(final_chunks: dict) -> str:
    '''
        description: simple function to make api call to LLM to generate final output based on reranked chunks.

        input: final_chunks, reranked chunks from cross-encoder step (TODO)
        output: final response (str)
    '''

    pass

if __name__ == "__main__":
    initial_chunks = bi_encoder_rank(PROCESSED_QUERY)
    final_chunks = cross_encoder_rerank(initial_chunks, PROCESSED_QUERY)
    final_output = output_generation(final_chunks)
    print(final_output)