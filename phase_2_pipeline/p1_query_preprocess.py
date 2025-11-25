# query preprocess module
from pydantic import BaseModel, Field
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase_2_pipeline.lib.gemini_client import get_gemini_client

class ProcessedQuery(BaseModel):
    query: str = Field(description="query with additional information using original query.")
    collection: str = Field(description="which collection this query should choose data from.")

# TODO modify this query to run the preprocessing function and see output in terminal
TEST_QUERY = "I am considering buying the Sony XM4 headphones and I had some questions before I buy them. What is the battery life on these, what charger do they support (usb c?), is there any control on the noise cancelling, is there an option to use it wired, what color options are there?"

SYS_PROMPT = '''You are tasked with taking prompts from users that are used to retrieve data from a RAG system. Improve the prompt by adding keywords,
restructuring it, and adding useful information using the following knowledge of the RAG vector storage:

The vector storage has 5 collections: ["camera_data", "displays_data", "headphone_data", "laptop_data", "phone_data"], determine which collection
would be best to read from to answer the users query. 

Each collection includes data in PDF, HTML, JSON, and TXT format. 

Here is some information on each collection:
camera_data: Canon Manual: https://www.usa.canon.com/support/p/eos-60d#idReference%3Dmanuals, Sony Manual: https://www.sony.com/electronics/support/cameras-camcorders-digital-cameras/manuals
Product Info: https://www.techradar.com/news/best-camera#section-the-best-value-camera-for-photography

displays_data: samsung tv manuals from https://www.samsung.com/us/support/downloads/?model=N0002200, headphone_data: downloaded manuals from: https://www.sony.com/electronics/support/audio-video-headphones/manuals 
manuals from: https://www.bose.co.uk/en_gb/support/products/bose_headphones_support/bose_in_ear_headphones_support/soundsport-free-wireless/manuals_downloads.html, https://support.bose.com/s/product/bose-quietcomfort-headphones/01t8c00000OydL4AAJ?language=en_US 
article: https://www.pcmag.com/picks/the-best-headphones?test_uuid=03iF1uOjHbmoZSTXr58OMhT&test_variant=A

laptop_data: HTML and PDF manuals Apple, HP, Dell, ASUS, Lenovo, Acer

phone_data: 1. Apple manuals, specs, downloads (HTML): https://support.apple.com/en-us/docs 
2. Google Pixel Guidebooks (HTML): https://guidebooks.google.com/pixel | Google Pixel Tech Specs (HTML): https://support.google.com/pixelphone/answer/7158570?hl=en#zippy=%2Cpixel%2Cpixel-pro%2Cpixel-pro-xl%2Cpixel-pro-fold 
3. Samsung User Guide and Repaire Guide (PDFs): https://www.samsung.com/uk/support/user-manuals-and-guide/ i.e. https://downloadcenter.samsung.com/content/UM/202509/20250926170639889/SM-A556_A566_A356_A366_UG_EU_16_Eng_Rev.1.0_250917.pdf 
4. Motorola User Guide (PDFs): https://en-us.support.motorola.com Note: After clicking on a phone, need to search ""pdf"" in search box to find the PDFs of the User Guides
5. OnePlus User Manual: https://service.oneplus.com/ee/user-manual#/

Return me a JSON object with "query" which is the improved query you make using their original query and another key "collection" for which collection to read from.
'''

def query_preprocess(raw_user_query: str) -> str:
    '''
        description: takes raw user query and adds information using LLM to potentially make the query better and help the RAG downstream.

        input: raw user query.
        output: processed user query.
    '''
    client = get_gemini_client()

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

    query_dict = json.loads(response.text)
    return query_dict

if __name__ == "__main__":
    print(query_preprocess(TEST_QUERY))