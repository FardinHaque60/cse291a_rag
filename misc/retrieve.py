# run file to retrieve an item by its id
import os
import sys
from qdrant_client import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_1_pipeline.client import get_client

# TODO update IDs for what items to retrieve 
ID = [
        ["eba3e997-29cd-4f19-8d13-be25399fa5a5",
        "f822f12a-241d-41ae-9b61-99c44173bdd0",],

        ["c328beae-cb80-4346-a440-58248c69f799",
        "4d71910b-77bf-4d2b-aa91-797a21592f8c",
        "7ee4e496-a112-439c-b4db-73190466f0f0",],

        ["1e4ffe29-e689-4389-9d06-804e9c6e8027",
        "a260cae8-7cf2-4752-83e2-e470e3e7773f",
        "cb71e926-cf66-41a7-83b3-1753daaf5dd1",
        "434d0b69-4d88-45ee-98d8-dd6d55615eff"],

        ["65702a47-1565-4f0d-8899-a7c7d51255e0",
        "ace0f57a-daaa-4fac-b8fb-764e893852d9",
        "7de62481-9323-4cb1-9c56-a3f6e98f9fbc",
        "80df52ff-5ddb-46ee-9aa6-695dc4c0bdab",],

        ["9d98e5a5-0c68-4c87-afed-348cc6d521e7",
        "5ac1aa0e-b5b6-4037-9d9d-1ee32f9b55d3",
        "24fd830a-8d1f-47a9-b923-2e296e29fdfe",
        "06f33908-b700-4957-9fce-b12217a5e980",
        "d41f32d8-24c2-4e58-b65c-28992229fe0f",
        "c20e366b-47a9-452e-9a7d-5196e4e9407a"]
    ]

categories = ["headphone_data", "laptop_data", "phone_data", "camera_data", "displays_data"]
actual_ids = []

client = get_client()

for i, category in enumerate(categories):
    # Example for a keyword index
    client.delete_payload_index(category, "source_file")

    client.create_payload_index(
        collection_name=category,
        field_name="source_file",
        field_schema="keyword"
    )

    client.create_payload_index(
        collection_name=category,
        field_name="page",
        field_schema="integer"
    )

    responses = client.retrieve(
                collection_name="production_data",
                ids=ID[i],            # or a list of IDs
                with_payload=True,
                with_vectors=False
            )
    
    for resp in responses:
        payload = resp.payload
        source_file = payload.get("source_file")
        page = payload.get("page")
        if source_file:
            query = {
                "source_file": source_file
            }
            if page is not None:
                query["page"] = page

            # Provide a dummy query_vector since it's required, but we only want to filter by payload
            # For example, use a zero vector of the correct dimension (e.g., 768)
            dummy_vector = [0.0] * 384  # Adjust dimension as needed for your collection

            # Build the 'must' list without None values
            must_conditions = [
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file)
                )
            ]
            if page is not None:
                must_conditions.append(
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=page)
                    )
                )

            filter_condition = models.Filter(
                must=must_conditions
            )

            results = client.search(
                collection_name=category,
                query_vector=dummy_vector,
                query_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )
            # Remove None from filter if page is None
            if results is not None:
                actual_ids.extend([r.payload.get("id") for r in results])

for id in actual_ids:
    with open("retrieved_responses2.txt", "a") as f:
        f.write(str(id) + "\n\n")