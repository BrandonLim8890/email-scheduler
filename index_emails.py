from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
from dotenv import load_dotenv
from utils import connect_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

load_dotenv(override=True)

chroma_path = "./chroma_store"
collection_name = "emails_enhanced"

embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])

llm = connect_model(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    model=os.getenv("CHAT_MODEL_NAME"),
)


def enhance_document(document: Document) -> Document:
    """Enhance a document with LLM-generated supplementary information."""
    enhancement_prompt = ChatPromptTemplate.from_template(
        """
    Please analyze the following email text and provide:
    
    1. 3-4 main topics covered in the email, including dates, times, locations, people, and entities mentioned.
    2. 3-4 questions this email could answer, especially related to scheduling (i.e., "When is the meeting?")
    
    Format your response with clear headings for each section. Be concise and to the point.
    
    EMAIL:
    {text}
    """
    )

    enhancement_chain = enhancement_prompt | llm | StrOutputParser()

    enhancements = enhancement_chain.invoke({"text": document.page_content})
    enhanced_content = f"""
ORIGINAL EMAIL:
{document.page_content}

ENHANCEMENTS:
{enhancements}
"""
    enhanced_doc = Document(page_content=enhanced_content, metadata=document.metadata)

    return enhanced_doc


def enhance_documents_batch(
    documents: List[Document], batch_size: int = 5
) -> List[Document]:
    enhanced_documents = []

    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch = documents[i:end_idx]

        print(f"Enhancing documents {i} to {end_idx-1} of {len(documents)}")

        for doc in batch:
            enhanced_doc = enhance_document(doc)
            enhanced_documents.append(enhanced_doc)

    return enhanced_documents


def collection_exists_with_docs():
    try:
        import chromadb

        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()

        for collection in collections:
            if collection == collection_name:
                chroma_collection = client.get_collection(collection_name)
                return chroma_collection.count() > 0
        return False
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


def main():
    if not collection_exists_with_docs():
        print(f"Creating new enhanced collection '{collection_name}'")

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

        enhanced_documents = enhance_documents_batch(documents, batch_size=5)

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_path,
        )

        indexing_batch_size = 50
        for i in range(0, len(enhanced_documents), indexing_batch_size):
            end_idx = min(i + indexing_batch_size, len(enhanced_documents))
            print(f"Adding documents {i} to {end_idx-1} of {len(enhanced_documents)}")
            vector_store.add_documents(documents=enhanced_documents[i:end_idx])

        print(
            f"Enhanced documents successfully indexed in collection '{collection_name}'"
        )
    else:
        print(
            f"Collection '{collection_name}' already exists with documents. Skipping indexing."
        )


if __name__ == "__main__":
    main()
