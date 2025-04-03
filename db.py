import pandas as pd
import chromadb
from chromadb.config import Settings
from pprint import pprint


df = pd.read_csv("emails/mann_meeting_related_2001_21.csv")
documents = df["content"].astype(str).tolist()
ids = [f"email_{i}" for i in range(len(df))]


client = chromadb.Client(Settings(persist_directory="./chroma_store"))
collection = client.get_or_create_collection(name="emails")
metadatas = df[["From", "To", "Subject", "Date"]].astype(str).to_dict(orient="records")

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Collection count:", collection.count())

results = collection.query(
    query_texts=["this is about a meeting"], 
    n_results=3 
)

pprint(results)