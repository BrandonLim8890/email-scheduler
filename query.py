import os

from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver

from utils import connect_model

load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])

vector_store = Chroma(
    collection_name="emails",
    embedding_function=embeddings,
    persist_directory="./chroma_store",
)


llm = connect_model(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"],
    model=os.environ["CHAT_MODEL_NAME"],
)

prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test_thread"}}

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ["quit", "exit", "q"]:
            break

        result = graph.invoke({"question": query}, config=config)
        print("\nAnswer:", result["answer"])

        print("\nSources:")
        context_docs = retrieve({"question": query})["context"]
        for i, doc in enumerate(context_docs[:3]):
            print(f"\nSource {i+1}:")
            print(f"From: {doc.metadata.get('From', 'Unknown')}")
            print(f"Subject: {doc.metadata.get('Subject', 'Unknown')}")
            print(f"Date: {doc.metadata.get('Date', 'Unknown')}")
