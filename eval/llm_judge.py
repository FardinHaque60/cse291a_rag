import json
from google import genai
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

INPUT_FILE = "eval/out/20251204_135015_metrics_phase2_50_biencoder_source_file_w_llm_responses.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYS_PROMPT = """
You are a judge that grades the response of a RAG system based on a given query. Your task is to evaluate the response in term of relevance to the user query on a scale from 1 to 5. 
Follow the criteria descriptions for each score closely when determining a grade.
<criteria>
1 - Poor: The response is completely irrelevant or incorrect. This should only be given for the most useless response with close to 0% relevance of the query. 
2 - Fair: The response has some relevance but lacks accuracy or completeness. This should be used for responses with around 25% relevance to the query. 
3 - Good: The response is generally accurate but may miss some details. This should be used for responses with around 50% relevance to the query. 
4 - Very Good: The response is accurate and covers most of the important details. This should be used for responses with around 75% relevance to the query. 
5 - Excellent: The response is highly accurate, comprehensive, and directly addresses the user query using useful information. Use this for the best responses which completely and correctly answer the query.
</criteria>
<instruction>
The user query will be found in the <user_query> tags and the response will be found in the <response> tags. 
IMPORTANT: Make sure that you give a reasoning for your grade in 1 line in addition to the grade.
Make sure you use the following output format 
</instruction> 
<output_format>
Reasoning: [1 sentence validating the response and reasoning on the rating]
Rating: [NUMBER] 
</output_format> 
"""

def rate_llm_responses(input_file, fw):
    # Load JSON list from file
    with open(input_file, "r") as f:
        data = json.load(f)

    data.pop() # remove the last element which has final average metrics

    # Configure Gemini API (ensure your API key is set in environment or here)
    client = genai.Client(api_key=GEMINI_API_KEY)

    for item in tqdm(data, desc="rating prompts"):
        query = item.get("prompt")
        llm_response = item.get("llm_response")
        query_for_llm = """
        <user_query>
        {query}
        </user_query>
        <response>
        {response}
        </response>
        """.format(query=query, response=llm_response)
        print("Running Query ")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "model", "parts": [{"text":SYS_PROMPT}]},
                {"role": "user", "parts": [{"text":query_for_llm}]},
            ],
        )
        print(f"Received Rating: {response.text}")
        fw.write(f"Prompt: {query}\nResponse: {llm_response}\nGemini Rating: {response.text}\n{'-'*40}\n")

if __name__ == "__main__":
    with open("eval/out/llm_judge_ratings.txt", "w") as fw:
        rate_llm_responses(INPUT_FILE, fw)