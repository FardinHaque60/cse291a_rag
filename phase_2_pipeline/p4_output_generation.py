# final output generation given final chunks
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_2_pipeline.p2_bi_encoder_rank import bi_encoder_rank
from phase_2_pipeline.p3_cross_encoder_rerank import cross_encoder_rerank
from phase_2_pipeline.lib.gemini_client import get_gemini_client
from phase_2_pipeline.lib.constants import LLM_CHUNKS
import pprint

SYS_PROMPT = '''You are tasked with generating the final response in a RAG system.
Your task is to generate a coherent and thoughtfully crafted response that answers the query correctly in 3-5 lines based on the context provided. 
The user query is given within the <user_query> tag and the relevant contexts are given within the <context> tag. 
Only use the given context to ground your responses, do not use outside references or knowledge to answer the user query.
'''

# TODO modify processed prompt passed into bi-encoder -> cross encoder, so chunk ranks can be used
PROCESSED_QUERY = {
    "query": "What are the best headphones according to the new york times?",
    "collection": "production_data",
}

# uses top 5 chunks for final llm response generation
def output_generation(final_chunks: list, query) -> str:
    '''
        description: simple function to make api call to LLM to generate final output based on reranked chunks.

        input: final_chunks, reranked chunks from cross-encoder step
        output: final response (str)
    '''
    client = get_gemini_client()

    chunk_text = []
    chunk_list = []
    chunk_count = 0
    for chunk in final_chunks:
        chunk_list.append(chunk[0])
        if chunk_count < LLM_CHUNKS:
            chunk_text.append(chunk[0].payload.get("text")[:500]) # TODO change to text once data is uploaded
        chunk_count += 1

    query_for_llm = """<user_query>
    {query}
    </user_query>
    <context>
    {context}
    </context>
    """.format(query=query, context="\n".join(chunk_text))

    with open("query_for_llm.txt", "w") as f:
        pprint.pprint(query_for_llm, stream=f)

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[
            {"role": "model", "parts": [{"text":SYS_PROMPT}]},
            {"role": "user", "parts": [{"text":query_for_llm}]},
        ],
    )

    return response.text, chunk_list
    # return "hi", [chunk[0] for chunk in final_chunks]

if __name__ == "__main__":
    initial_chunks = bi_encoder_rank(PROCESSED_QUERY)
    final_chunks = cross_encoder_rerank(initial_chunks, PROCESSED_QUERY)
    final_output = output_generation(final_chunks, PROCESSED_QUERY["query"])
    print(final_output)