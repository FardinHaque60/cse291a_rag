import json
from google import genai
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

INPUT_FILE = "eval/out/20251126_163345_metrics_phase2_llm_responses.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYS_PROMPT = """
You are a strict judge. Your task is to evaluate the response based on the user query on a likert scale from 1 to 5.
1 - Poor: The response is irrelevant or incorrect.
2 - Fair: The response has some relevance but lacks accuracy or completeness.
3 - Good: The response is generally accurate but may miss some details.
4 - Very Good: The response is accurate and covers most of the important details.
5 - Excellent: The response is highly accurate, comprehensive, and directly addresses the user query using useful information.

The users query will be found in the <user_query> tags and the response will be found in the <response> tags.
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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "model", "parts": [{"text":SYS_PROMPT}]},
                {"role": "user", "parts": [{"text":query_for_llm}]},
            ],
        )

        fw.write(f"Prompt: {query}\nResponse: {llm_response}\nGemini Rating: {response.text}\n{'-'*40}\n")

if __name__ == "__main__":
    with open("eval/out/llm_judge_ratings.txt", "w") as fw:
        rate_llm_responses(INPUT_FILE, fw)