# final output generation given final chunks
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_2_pipeline.p2_bi_encoder_rank import bi_encoder_rank
from phase_2_pipeline.p3_cross_encoder_rerank import cross_encoder_rerank
from phase_2_pipeline.lib.gemini_client import get_gemini_client

SYS_PROMPT = '''You are tasked with generating the final response in a RAG system. Given a query and the raw text from several of the highest ranked chunks, 
generate a coherent and thoughtfully crafted response that answers the query correctly in a few sentences.
'''

# TODO modify processed prompt passed into bi-encoder -> cross encoder, so chunk ranks can be used
PROCESSED_QUERY = {
    "query": "What are the best headphones according to the new york times?",
    "collection": "production_data",
}

def output_generation(final_chunks: list, query) -> str:
    '''
        description: simple function to make api call to LLM to generate final output based on reranked chunks.

        input: final_chunks, reranked chunks from cross-encoder step (TODO)
        output: final response (str)
    '''
    client = get_gemini_client()

    query_w_chunks = query + "retrieved chunks: \n"
    chunk_list = []
    for chunk in final_chunks:
        chunk_list.append(chunk[0])
        query_w_chunks += chunk[0].payload.get("source_file") # TODO change to text once data is uploaded

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {"role": "model", "parts": [{"text":SYS_PROMPT}]},
            {"role": "user", "parts": [{"text":query_w_chunks}]},
        ],
    )

    return response.text, chunk_list

if __name__ == "__main__":
    initial_chunks = bi_encoder_rank(PROCESSED_QUERY)
    final_chunks = cross_encoder_rerank(initial_chunks, PROCESSED_QUERY)
    final_output = output_generation(final_chunks, PROCESSED_QUERY["query"])
    print(final_output)