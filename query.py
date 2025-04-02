import chromadb
from chromadb.config import Settings
from pprint import pprint

client = chromadb.Client(Settings(persist_directory="./chroma_store"))
collection = client.get_or_create_collection("emails")

results = collection.query(
    query_texts=["There is a meeting sunday morning"],
    n_results=3
)

pprint(results)
