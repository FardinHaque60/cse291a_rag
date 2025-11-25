from qdrant_client import QdrantClient
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase_2_pipeline.lib.constants import QDRANT_URL, QDRANT_KEY

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_KEY,
)

def get_qdrant_client() -> QdrantClient:
    return client