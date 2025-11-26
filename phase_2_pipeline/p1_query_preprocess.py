# query preprocess module
from pydantic import BaseModel, Field
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase_2_pipeline.lib.gemini_client import get_gemini_client

class ProcessedQuery(BaseModel):
    query: str = Field(description="query modified with LLM processing")
    collection: str = Field(description="which collection this query should choose data from.")

# TODO modify this query to run the preprocessing function and see output in terminal

# Running it with a smaller test query to see how effectively gemini adds information
TEST_QUERY = "What are the best computers for gaming?"

SYS_PROMPT = '''
You are tasked with preprocessing queries from users that are attempting to retrieve data from a RAG system. 
Improve the prompt by adding keywords, restructuring it, and adding useful information that other LLMs and RAGS would be able to efficiently use.


The structure of the RAG vector storage is as follows:
5 collections: ["camera_data", "displays_data", "headphone_data", "laptop_data", "phone_data"]

Each collection includes data in PDF, HTML, JSON, and TXT format. 

Here is some information on each collection:
camera_data: Manufacturer (Sony/Canon,etc.) Manuals, User Guides, Opinion Articles on Cameras

displays_data: TV and Monitor Manuals from Manufacturers, Opinion Articles on TVs and Monitors

headphone_data: Manufacturer Manuals on Headphones (Sony, Bose, Sennheiser, JBL, etc.), Reviews and Articles on Headphones

laptop_data: HTML and PDF manuals Apple, HP, Dell, ASUS, Lenovo, Acer

phone_data: 1. Apple manuals, specs, downloads (HTML):
2. Google Pixel Guidebooks (HTML) | Google Pixel Tech Specs (HTML)
3. Samsung User Guide and Repaire Guide (PDFs)
4. Motorola User Guide (PDFs)
5. OnePlus User Manual
6. Opinion Articles on Phones

Return me a JSON object with the first key "query", which is the improved processed query and another key "collection", which indicates which collection we should pull data from to answer the query.

The JSON should follow the structure below: 
{
  "query": "...",
  "collection": "..."
}
'''

def query_preprocess(raw_user_query: str) -> str:
    '''
        description: takes raw user query and adds information using LLM to potentially make the query better and help the RAG downstream.

        input: raw user query.
        output: processed user query.
    '''
    client = get_gemini_client()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                {"role": "model", "parts": [{"text":SYS_PROMPT}]},
                {"role": "user", "parts": [{"text":raw_user_query}]},
            ],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ProcessedQuery.model_json_schema(),
            },
        )
    except Exception as e:
        print(f"Error during query preprocessing: {e}")
        raise e

    # response_json_schema does not guarantee valid json, so throw error if invalid
    try:
        query_dict = json.loads(response.text)
        return query_dict
    except exception as e:
        print(f"Error parsing JSON response: {e}")
        raise e

if __name__ == "__main__":
    print(query_preprocess(TEST_QUERY))