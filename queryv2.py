import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from sentence_transformers import CrossEncoder

from utils import connect_model, create_google_calendar_event

load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
reranker = CrossEncoder("BAAI/bge-reranker-large", device="mps")

vector_store = Chroma(
    collection_name="emails_enhanced_full_inbox",
    embedding_function=embeddings,
    persist_directory="./chroma_store",
)

llm = connect_model(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    model=os.getenv("CHAT_MODEL_NAME"),
)


@tool
def add_to_calendar(
    title: str,
    start_time: str,
    end_time: str,
    location: str = "",
    description: str = "",
):
    """Add a new event to Google Calendar. Time format: 'YYYY-MM-DDTHH:MM:SS' (ISO 8601)."""
    event = create_google_calendar_event(
        title=title,
        start_time=start_time,
        end_time=end_time,
        location=location,
        description=description,
    )
    return f"Event '{event.get('summary')}' created on calendar from {start_time} to {end_time} at {location}."


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Searches the email vector store for content relevant to the given query and reranks the results using a BGE reranker."""

    retrieved_docs = vector_store.similarity_search(query, k=25)

    rerank_inputs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(rerank_inputs)

    scored_docs = list(zip(retrieved_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, _ in scored_docs[:5]]

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in top_docs
    )

    return serialized, top_docs


tools = ToolNode([retrieve, add_to_calendar])


memory = MemorySaver()

config = {"configurable": {"thread_id": "abc123"}}

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

system_msg = SystemMessage(
    content="You are an assistant for scheduling calendar meetings and appointments. The year is 2001. "
    "The user's name is Kay Mann. Use the tools available to you to respond appropriately. "
    "Be concise and polite."
)

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ["quit", "exit", "q"]:
            break

        last_msg = None

        for event in agent_executor.stream(
            {"messages": [system_msg, {"role": "user", "content": query}]},
            stream_mode="values",
            config=config,
        ):
            last_msg = event["messages"][-1]
        last_msg.pretty_print()
