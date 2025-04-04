from langchain_openai import ChatOpenAI


def connect_model(api_key, base_url, model, temperature=0.4):
    llm = ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, temperature=temperature
    )
    return llm
