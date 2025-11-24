# helper file to make connection with qdrant client
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
# embeddings model from qdrant, model should not need changing
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key,
)

def get_client():
    return client