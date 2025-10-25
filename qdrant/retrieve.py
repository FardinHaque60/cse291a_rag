# run file to retrieve an item by its id
from client import get_client

ID = [
      "bf8136f3-bc3b-4ef3-8fc1-a5a7545b49e0",
      "cf130982-22e2-4ecf-83f9-07c7a6245ba8",
      "0a258316-bbcd-4741-a3f7-d86db182e3dd",
      "fcb84c50-b470-4e82-b41d-e711dd8233b5",
      "08d3a8b6-5788-4edf-b40e-f8142c42499f",
      "0568bb8d-7cb3-41ed-aed9-a68695e31e70",
      "4c15d349-c622-4d9f-95bc-978b9c99fb22",
      "db7de967-9c67-4b4b-b04d-427f7eacab1a",
      "5aa4c538-107c-417e-bfde-947d4a7814e7",
      "ebc89905-61a2-423d-9771-b8323bc4e74a"
    ]

client = get_client()

responses = client.retrieve(
            collection_name="production_data",
            ids=ID,            # or a list of IDs
            with_payload=True,
            with_vectors=False
        )

for response in responses:
    print(response)
    print()