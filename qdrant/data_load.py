# edit this file with data to load and run it to push it to the qdrant cluster: python qdrant/data_load.py
import os
import uuid
from client import get_client, EMBEDDING_MODEL_NAME
from qdrant_client import models
from pypdf import PdfReader
from fastembed import TextEmbedding
from bs4 import BeautifulSoup
import json

## ------ TODO ------ 
# modify fields based on location of data, name of collection to store it, and data types
INPUT_DIRECTORY = os.getcwd() + "/data/HTML"
# if entered collection name that exists, data will be added to collection, if collection doesnt exist, then it will be created with data
QDRANT_COLLECTION_NAME = "headphone_data"
# valid formats are PDF, HTML, TXT, and JSON
DATA_FORMAT = "HTML"

def process_pdf_from_directory():
    """
    Scans a directory for PDF files, extracts text, and splits it into chunks.
    
    Returns:
        A list of dictionaries, where each dictionary contains the source filename
        and a specific chunk of text.
    """
    all_text_chunks = []

    print(f"Scanning for PDF files in '{INPUT_DIRECTORY}'...")
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(INPUT_DIRECTORY, filename)
            try:
                reader = PdfReader(file_path)
                full_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                
                # split by paragraphs.
                chunks = [chunk.strip() for chunk in full_text.split('\n\n') if chunk.strip()]
                
                for text_chunk in chunks:
                    all_text_chunks.append({
                        "source_file": filename,
                        "text": text_chunk
                    })
                print(f"  - Extracted {len(chunks)} text chunks from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def process_html_from_directory():
    """
    Scans a directory for .html files, extracts the visible text, and splits it into chunks.
    
    Requires `pip install beautifulsoup4`.
    
    Returns:
        A list of dictionaries, where each dictionary contains the source filename,
        the page title, and a specific chunk of text.
    """
    all_text_chunks = []

    print(f"Scanning for .html files in '{INPUT_DIRECTORY}'...")
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.lower().endswith((".html", ".htm")):
            file_path = os.path.join(INPUT_DIRECTORY, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                
                # Extract page title for metadata, if it exists
                page_title = soup.title.string if soup.title else "No Title"

                # Get all human-readable text from the body
                # .get_text() is powerful; it strips tags and combines text.
                # The separator ensures paragraphs are spaced, making chunking reliable.
                full_text = soup.body.get_text(separator='\n\n', strip=True)

                # Split by paragraphs
                chunks = [chunk.strip() for chunk in full_text.split('\n\n') if chunk.strip()]
                
                for text_chunk in chunks:
                    all_text_chunks.append({
                        "source_file": filename,
                        "title": page_title,
                        "text": text_chunk
                    })
                print(f"  - Extracted {len(chunks)} text chunks from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def process_txt_from_directory():
    """
    Scans a directory for .txt files, reads their content, and splits it into chunks.
    
    Returns:
        A list of dictionaries, where each dictionary contains the source filename
        and a specific chunk of text.
    """
    all_text_chunks = []

    print(f"Scanning for .txt files in '{INPUT_DIRECTORY}'...")
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(INPUT_DIRECTORY, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                # Split by paragraphs (double newline). This is a robust way to chunk text files.
                chunks = [chunk.strip() for chunk in full_text.split('\n\n') if chunk.strip()]
                
                for text_chunk in chunks:
                    all_text_chunks.append({
                        "source_file": filename,
                        "text": text_chunk
                    })
                print(f"  - Extracted {len(chunks)} text chunks from '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def process_json_from_directory():
    all_text_chunks = []

    print(f"Scanning for .json files in '{INPUT_DIRECTORY}'...")
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(INPUT_DIRECTORY, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Ensure data is a list to process records uniformly
                if not isinstance(data, list):
                    data = [data] # Treat a single object as a list with one item

                records_processed = 0
                for record in data:
                    if not isinstance(record, dict):
                        # Skip non-object items in the list if the JSON is mixed
                        continue
                    
                    payload = record.copy()
                    payload["source_file"] = filename
                    text_to_embed = json.dumps(record, indent=2)
                    payload["text"] = text_to_embed
                    
                    all_text_chunks.append(payload)
                    records_processed += 1
                
                print(f"  - Extracted text from {records_processed} records in '{filename}'.")

            except Exception as e:
                print(f"  - ERROR: Failed to process '{filename}': {e}")
                
    return all_text_chunks

def main(format):
    """
    Main function to orchestrate the PDF processing and uploading workflow.
    """
    client = get_client()
    
    print(f"\nSetting up Qdrant collection: '{QDRANT_COLLECTION_NAME}'")

    # check if collection exists, if not create it
    try:
        # Check if the collection already exists
        client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Adding data to it.")
    except Exception as e:
        # If the collection does not exist, an exception is thrown
        print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist. Creating it now.")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
        )
        print("Collection created successfully.")

    print("Collection setup complete.")

    # --- 3. Read and Process Local PDFs ---
    document_chunks = []
    match format:
        case "PDF":
            document_chunks = process_pdf_from_directory()
        case "HTML":
            document_chunks = process_html_from_directory()
        case "TXT":
            document_chunks = process_txt_from_directory()
        case "JSON":
            document_chunks = process_json_from_directory()

    if not document_chunks:
        print("\nNo documents found or processed. Exiting.")
        return

    # Convert Text to Vectors (Embeddings) ---
    print(f"\nInitializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model initialized.")

    # Prepare the list of texts to be embedded.
    texts_to_embed = [item["text"] for item in document_chunks]
    
    print(f"Creating embeddings for {len(texts_to_embed)} text chunks...")
    embeddings_result = embedding_model.embed(texts_to_embed)
    print("Embeddings created successfully.")
    
    # upload to qdrant
    print(f"\nUploading {len(document_chunks)} points to Qdrant...")
    
    points_to_upload = []
    for vector, chunk_data in zip(embeddings_result, document_chunks):
        points_to_upload.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=chunk_data
            )
        )

    client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points_to_upload,
        batch_size=256
    )

    print("Upload complete!")
    print(f"\nWorkflow finished. Your data is now in the '{QDRANT_COLLECTION_NAME}' collection.")

if __name__ == "__main__":
    main(DATA_FORMAT)
