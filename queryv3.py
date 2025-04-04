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

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = '\n\n'.join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}" for doc in retrieved_docs) 
    )

    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    router_llm = llm.bind(stop=["\n"])
    should_retrieve = router_llm.invoke(
        state["messages"] + [{"role": "system", "content": "Is the above question about looking at a schedule/meetings/calendar? Answer 'YES' or 'NO'"}]
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

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for scheduling calendar meetings and appointments. The user's name is Kay Mann. For every query, you will be given context from the user's email inbox. The context includes the email content as well as metadata such as the subject and date of the email, as well as senders and recipients. Sometimes, the context will be relevant to the question and other times it will not. Any time the user asks about ANY meetings, appointments, or calendar-related events, use the retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Make sure you only provide real information."
        "\n\n"
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
