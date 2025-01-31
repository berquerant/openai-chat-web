from langchain_openai import ChatOpenAI


def chat(model: str = "gpt-4o", temperature: float = 0.7, timeout: float = 3.5) -> ChatOpenAI:
    """Return a new chat model."""
    return ChatOpenAI(model=model, temperature=temperature, timeout=timeout)
