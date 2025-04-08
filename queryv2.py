import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from sentence_transformers import CrossEncoder
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

from utils import connect_model

load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL_NAME"])
reranker = CrossEncoder("BAAI/bge-reranker-large", device="mps")

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
    llm_with_tools = llm.bind_tools([retrieve])
    system_message = SystemMessage(
        "You are an assistant for scheduling calendar meetings and appointments. "
        "ALWAYS use the retrieve tool to search through emails for specific meeting information. "
        "Use the retrieve tool for ANY question about meetings, appointments, or calendar information."
    )

    messages_with_system = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages_with_system)

    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for scheduling calendar meetings and appointments.\n"
        "The user's name is Kay Mann. For every query, you will be given context from the user's email inbox.\n"
        "The context includes the email content and additional information, as well as metadata such as the subject and date of the email, as well as senders and recipients.\n"
        "Use the retrieved context to answer the question.\n"
        "If you don't know the answer, just say that you don't know.\n"
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
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)


input_message = (
    "What meetings do I have on June 5?\n\n"
    "Once you get the answer, tell me where they are."
)

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() in ["quit", "exit", "q"]:
            break

        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            event["messages"][-1].pretty_print()
