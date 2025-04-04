import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Dict
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

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
conversation_prompt = client.pull_prompt("kay-mann")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_history: List[Dict]


def retrieve(state: State):
    question = state["question"]

    chat_history = state.get("chat_history", [])

    enhanced_query = question
    if chat_history:
        recent_exchanges = chat_history[-3:]
        history_context = " ".join(
            [
                f"{exchange['question']} {exchange['answer']}"
                for exchange in recent_exchanges
            ]
        )
        enhanced_query = f"{history_context} {question}"

    retrieved_docs = vector_store.similarity_search(enhanced_query)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    chat_history = state.get("chat_history", [])

    formatted_history = []
    for exchange in chat_history:
        formatted_history.append({"role": "human", "content": exchange["question"]})
        formatted_history.append({"role": "assistant", "content": exchange["answer"]})

    messages = conversation_prompt.invoke(
        {
            "question": state["question"],
            "context": docs_content,
            "chat_history": formatted_history,
        }
    )

    response = llm.invoke(messages)
    return {"answer": response.content}


def update_history(state: State):
    chat_history = state.get("chat_history", [])

    chat_history.append({"question": state["question"], "answer": state["answer"]})

    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return {"chat_history": chat_history}


graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("update_history", update_history)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "update_history")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test_thread"}}

if __name__ == "__main__":
    # Initialize empty chat history
    chat_history = []

    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ["quit", "exit", "q"]:
            break

        # Initial state includes the chat history
        initial_state = {"question": query, "chat_history": chat_history}

        # Run the graph
        result = graph.invoke(initial_state, config=config)

        # Update our local copy of chat history
        chat_history = result["chat_history"]

        print("\nAnswer:", result["answer"])
        print("\nSources:")
        context_docs = retrieve({"question": query, "chat_history": chat_history})[
            "context"
        ]
        for i, doc in enumerate(context_docs[:3]):
            print(f"\nSource {i+1}:")
            print(f"From: {doc.metadata.get('From', 'Unknown')}")
            print(f"Subject: {doc.metadata.get('Subject', 'Unknown')}")
            print(f"Date: {doc.metadata.get('Date', 'Unknown')}")
