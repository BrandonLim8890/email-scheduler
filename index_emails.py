from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(override=True)

chroma_path = "./chroma_store"
collection_name = "emails"

embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])


def collection_exists_with_docs():
    try:
        import chromadb

        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()

        for collection in collections:
            if collection == collection_name:
                # Check if collection has documents
                chroma_collection = client.get_collection(collection_name)
                return chroma_collection.count() > 0
        return False
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


if not collection_exists_with_docs():
    df = pd.read_csv("emails/mann_meeting_related_2001_21.csv")

    documents = []
    for i, row in df.iterrows():
        content = str(row["content"])
        metadata = {
            "From": str(row["From"]),
            "To": str(row["To"]),
            "Subject": str(row["Subject"]),
            "Date": str(row["Date"]),
            "id": f"email_{i}",
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )

    batch_size = 50
    for i in range(0, len(all_splits), batch_size):
        end_idx = min(i + batch_size, len(all_splits))
        print(f"Adding documents {i} to {end_idx} of {len(all_splits)}")
        vector_store.add_documents(documents=all_splits[i:end_idx])

    print(f"Documents successfully indexed in collection '{collection_name}'")
else:
    print(
        f"Collection '{collection_name}' already exists with documents. Skipping indexing."
    )
