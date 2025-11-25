# run pipeline runner

from p1_query_preprocess import query_preprocess
from p2_bi_encoder_rank import bi_encoder_rank
from p3_cross_encoder_rerank import cross_encoder_rerank
from p4_output_generation import output_generation

# TODO change query to pass into pipeline for unit testing
QUERY = "I am considering buying the Sony XM4 headphones and I had some questions before I buy them. What is the battery life on these, what charger do they support (usb c?), is there any control on the noise cancelling, is there an option to use it wired, what color options are there?"

# function called by phase_2_pipeline to run automated evals
def run_pipeline(query) -> str:
    '''
        description: takes raw user query and runs our RAG pipeline. 

        input: raw user query.
        output: final output generation string.
    '''

    padded_query = query_preprocess(query)
    initial_ranking = bi_encoder_rank(padded_query)
    final_rank = cross_encoder_rerank(initial_ranking, padded_query)
    # final_output = output_generation(final_rank)

    return final_rank

if __name__ == "__main__":
    print(run_pipeline(QUERY))