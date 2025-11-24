# phase 1
files in this directory are related to phase 1, using qdrant by itself to perform RAG.

## pipeline overview
1. load data from data sources to qdrant collection with `python phase_1_pipeline/data_load.py`
    - data is loaded by embedding the entire data sources and storing the vectorized version
    - PDFs are chunked by page, so each page is vectorized and stored. other data sources are saved whole though
2. run retrieval with `python phase_1_pipeline/inference.py`
    - performs cosine similarity search using embedded query with vector storage entries
    - returns retrieved chunks (w/ metadata information) and latency information