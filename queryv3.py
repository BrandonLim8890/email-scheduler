import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sentence_transformers import CrossEncoder

from utils import connect_model

reranker = CrossEncoder("BAAI/bge-reranker-large", device="mps")


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
        "You are an assistant for scheduling calendar meetings and appointments.\n"
        "The user's name is Kay Mann. For every query, you will be given context from the user's email inbox.\n"
        "The context includes the email content and additional information, as well as metadata such as the subject and date of the email, as well as senders and recipients.\n"
        "Use the retrieved context to answer the question.\n"
        "The output of the schedule should be in the following format:\n'Schedule:\n - {MMDD} {hh:mm}{AM/PM}: {event}\n - {MMDD} {hh:mm}{AM/PM}: {event}'\n"
        "If there is no schedule, then output:\n'Schedule:\nNo events found.'\n"
        "If you don't know the answer then output:\n'I don't know'.\n"
        "Use three sentences maximum and keep the answer concise.\n\n"
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
    if sys.stdin.isatty():
        input_source = iter(lambda: input("User (blank to quit): "), "")
    else:
        input_source = sys.stdin.read().strip().split("\n")

    for user_input in input_source:
        user_input = user_input.strip()
        if not user_input:
            break

        last_message = None

        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            last_message = step["messages"][-1]

        last_message.pretty_print()
