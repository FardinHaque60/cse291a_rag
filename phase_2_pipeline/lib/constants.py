import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # bi encoder
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" # cross encoder
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")
COLLECTION_NAME = "production_data"
RESULTS_COUNT = 10 # represents the number of initial search results to retrieve from Qdrant (used for bi encoder)