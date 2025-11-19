from .constants import QDRANT_URL, QDRANT_KEY
from qdrant_client import QdrantClient

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_KEY,
)

def get_qdrant_client() -> QdrantClient:
    return client