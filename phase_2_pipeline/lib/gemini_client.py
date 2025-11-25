import os
import sys
from google import genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_2_pipeline.lib.constants import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

def get_gemini_client():
    return client