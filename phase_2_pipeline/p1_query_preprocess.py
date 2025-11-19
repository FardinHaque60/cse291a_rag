# query preprocess module

# TODO modify this query to run the preprocessing function and see output in terminal
TEST_QUERY = ""

def query_preprocess(raw_user_query: str) -> str:
    '''
        description: takes raw user query and adds information using LLM to potentially make the query better and help the RAG downstream.

        input: raw user query.
        output: processed user query.
    '''
    pass

if __name__ == "__main__":
    print(query_preprocess(TEST_QUERY))