from email.message import Message
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.runnables.config import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from utils import connect_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from pprint import pprint
from sentence_transformers import CrossEncoder
import torch

reranker = CrossEncoder(
    "BAAI/bge-reranker-large", device="cuda" if torch.cuda.is_available() else "cpu"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)


load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])


vector_store = Chroma(
    collection_name="emails_enhanced",
    embedding_function=embeddings,
    persist_directory="./chroma_store",
)

llm = connect_model(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"],
    model=os.environ["CHAT_MODEL_NAME"],
)

graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve and rerank information related to a query using bge reranker."""

    retrieved_docs = vector_store.similarity_search(query, k=25)

    rerank_inputs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(rerank_inputs)

    scored_docs = list(zip(retrieved_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, score in scored_docs[:5]]

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in top_docs
    )

    return serialized, top_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    router_llm = llm.bind(stop=["\n"])
    should_retrieve = router_llm.invoke(
        state["messages"]
        + [
            {
                "role": "user",
                "content": "Is the above question asking about meetings, events, scheduling, appointments, or related? Answer 'YES' or 'NO'",
            }
        ]
    )
    pprint("SHOULD USE RETRIEVE?")
    pprint(should_retrieve.content)
    if "YES" in should_retrieve.content:
        llm_with_tools = llm.bind_tools([retrieve], tool_choice="required")
    else:
        llm_with_tools = llm.bind_tools([retrieve])

    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate answer"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(
        f"[EMAIL {i+1}]\nSubject: {msg.additional_kwargs.get('metadata', {}).get('subject', '')}\n"
        f"Date: {msg.additional_kwargs.get('metadata', {}).get('date', '')}\n"
        f"Content: {msg.content.strip()}"
        for i, msg in enumerate(tool_messages)
    )

    system_message_content = (
        "You are a helpful assistant whose job is to build a schedule based entirely on the user's emails. "
        "The user's name is Kay Mann.\n\n"
        "You will be given snippets of emails that may include subject lines, dates, senders, recipients, and body text. "
        "These emails may mention meetings, appointments, or events.\n\n"
        "Your task is to extract and list any events that are mentioned — especially ones with specific dates, times, participants, or locations. "
        "Assume that if an event is mentioned in an email the user received or was CC'd on, they are likely involved unless stated otherwise.\n\n"
        "**Only return a bulleted list of events. Do not explain, summarize, or add commentary.**\n"
        "**If there is no event found, respond with: 'According to your emails, there is nothing scheduled for that day.'**\n\n"
        "Use this format:\n"
        "- [Date, Time] — [Meeting/Event name or participants, location if known]\n\n"
        "Context:\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    print("\n\n==== SYSTEM PROMPT ====\n")
    print(system_message_content)

    return {"messages": [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
config = {"configurable": {"thread_id": "thread_id"}}

graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ["quit", "exit", "q"]:
            break

        for step in graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()
