import requests
import os
import sys
import uuid
from qdrant_client import models
from pypdf import PdfReader
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase_2_pipeline.lib.qdrant_client import get_qdrant_client
from phase_2_pipeline.lib.embedding_models import bi_encoder_model

# ---- TODO used for unit testing ----
# modify fields based on location of data, name of collection to store it, and data types
INPUT_DIRECTORY = os.getcwd() + "/data/temp/"
# valid formats are PDF, HTML, TXT, and JSON
DATA_FORMAT = "PDF"

SYS_PROMPT = '''Return me a json object with one key being 'summary' which is a 4-5 sentence summary of the text in the queries I pass in. For the summary,
you can jump straight into the summary, avoid filler words like "this text is about" or "the provided text", etc. MAKE SURE SUMMARIES are in ENGLISH. TRANSLATE 
IF NEEDED, I DONT WANT SUMMARIES IN ANY OTHER LANGUAGE.
For the second key, 'keywords', include a list of 5-8 keywords that best fit the text in the queries I pass in.'''

# returns json object from LLM with summary string and keywords list
def gen_metadata(chunk_text):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": chunk_text}
        ],
        "temperature": 0.3,
        "max_tokens": -1,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    structured_response = response.json()['choices'][0]['message']['content']
    structured_response = json.loads(structured_response)
    return structured_response

def process_pdf_from_directory(dir_path):
    """
    Scans a directory for PDF files, extracts text, and splits it into chunks by page.
    
    Returns:
        A list of dictionaries, where each dictionary contains the source filename
        and a specific chunk of text.
    """
    all_text_chunks = []

    print(f"Scanning for PDF files in '{dir_path}'...")
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(dir_path, filename)
            try:
                reader = PdfReader(file_path)
                chunks = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        chunks.append(page_text.strip())
                
                for i, text in enumerate(chunks):
                    all_text_chunks.append({
                        "source_file": filename,
                        "page": i+1,
                        "text": text 
                    })
                    if i > 50:
                        break
                print(f"  - Extracted {len(chunks)} text chunks from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def process_html_from_directory(dir_path):
    """
    Scans a directory for .html files, extracts the visible text, and splits it into chunks.
    
    Requires `pip install beautifulsoup4`.
    
    Returns:
        A list of dictionaries, where each dictionary contains the source filename,
        the page title, and a specific chunk of text.
    """
    all_text_chunks = []

    print(f"Scanning for .html files in '{dir_path}'...")
    for filename in os.listdir(dir_path):
        if filename.lower().endswith((".html", ".htm")):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                
                # Extract page title for metadata, if it exists
                page_title = soup.title.string if soup.title else "No Title"

                # Get all human-readable text from the body
                # .get_text() is powerful; it strips tags and combines text.
                # The separator ensures paragraphs are spaced, making chunking reliable.
                full_text = soup.body.get_text(separator='\n\n', strip=True)
                
                all_text_chunks.append({
                    "source_file": filename,
                    "title": page_title,
                    "text": full_text
                })
                print(f"  - Extracted from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def process_txt_from_directory(dir_path):
    """
    Scans a directory for .txt files, reads their content, and splits it into chunks.
    
    Returns:
        A list of dictionaries, where each dictionary contains the source filename
        and a specific chunk of text.
    """
    all_text_chunks = []

    print(f"Scanning for .txt files in '{dir_path}'...")
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                all_text_chunks.append({
                    "source_file": filename,
                    "text": full_text
                })
                print(f"  - Extracted from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def process_json_from_directory(dir_path):
    '''
    expects data represented as a single json object
    '''
    all_text_chunks = []

    print(f"Scanning for .json files in '{dir_path}'...")
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Ensure data is one json object
                if not isinstance(data, dict):
                    data = {"data": data}

                payload = {}
                payload["source_file"] = filename
                payload["text"] = json.dumps(data, indent=2)
                all_text_chunks.append(payload)
                
                print(f"  - Extracted text from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def upload_to_qdrant(chunks, collection_name):
    """
    Main function to orchestrate the PDF processing and uploading workflow.
    """
    client = get_qdrant_client()
    
    print(f"\nSetting up Qdrant collection: '{collection_name}'")

    # check if collection exists, if not create it
    try:
        # Check if the collection already exists
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists. Adding data to it.")
    except Exception as e:
        # If the collection does not exist, an exception is thrown
        print(f"Collection '{collection_name}' does not exist. Creating it now.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
        )
        print("Collection created successfully.")

    print("Collection setup complete.")

    # Convert Text to Vectors (Embeddings) ---
    embedding_model = bi_encoder_model()
    
    print(f"Creating embeddings for {len(chunks)} text chunks...")
    embeddings_result = embedding_model.embed([item["text"] for item in chunks])
    print("Embeddings created successfully.")
    
    # upload to qdrant
    print(f"\nUploading {len(chunks)} points to Qdrant...")
    
    points_to_upload = []
    for vector, chunk_data in zip(embeddings_result, chunks):
        # del chunk_data["text"] # remove raw text from metadata
        points_to_upload.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=chunk_data
            )
        )

    client.upload_points(
        collection_name=collection_name,
        points=points_to_upload,
        batch_size=256
    )

    print("Upload complete!")

if __name__ == "__main__":
    data_sources = ["camera_data", "displays_data", "headphone_data", "laptop_data", "phone_data"]
    data_formats = ["PDF", "HTML", "TXT", "JSON"]

    data_sources = ["headphone_data/articles", "headphone_data/manuals", "laptop_data/HTML", "laptop_data/PDF"]

    for data_source in data_sources:
        for data_format in data_formats:
            dir_path = os.getcwd() + f"/data/{data_source}/"
            print(f"\nProcessing data from '{dir_path}' in format '{data_format}'...")
            if data_format == "PDF":
                data = process_pdf_from_directory(dir_path)
            elif data_format == "HTML":
                data = process_html_from_directory(dir_path)
            elif data_format == "TXT":
                data = process_txt_from_directory(dir_path)
            elif data_format == "JSON":
                data = process_json_from_directory(dir_path)

            print(f"Generated summaries for data from '{data_source}' in format '{data_format}':")
            for item in tqdm(data, desc="generating keywords & summaries"):
                metadata = gen_metadata(item['text'][:2000])
                if "summary" in metadata:
                    item['summary'] = metadata['summary']
                if "keywords" in metadata:
                    item["keywords"] = metadata["keywords"]

            collection_name = data_source.split("/")
            upload_to_qdrant(data, collection_name[0])

        print(f"\nAdding data to '{data_source}' collection completed.")