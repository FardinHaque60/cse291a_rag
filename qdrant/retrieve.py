# run file to retrieve an item by its id
from client import get_client

ID = "eba3e997-29cd-4f19-8d13-be25399fa5a5"

client = get_client()

response = client.retrieve(
    collection_name="production_data",
    ids=[ID],            # or a list of IDs
    with_payload=True,
    with_vectors=False
)

print(response)