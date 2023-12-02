import re

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun


class Agent:
    executor: AgentExecutor

    def __init__(self, executor: AgentExecutor):
        self.executor = executor

    def send(self, message: str) -> str:
        # https://github.com/langchain-ai/langchain/issues/1358#issuecomment-1486132587
        try:
            return self.executor.run(message).lstrip()
        except ValueError as e:
            matched = re.search(r"Could not parse LLM output: (.+)", str(e))
            if matched:
                return matched.group(1)
            raise


def new(chat_model: str, temperature: float, verbose: bool) -> Agent:
    """Return new `Agent`."""
    searcher = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="duckduckgo-search",
            func=searcher.run,
            description="useful for when you need to search for latest information in web",
        ),
    ]
    # mypy causes Unexpected keyword argument "model_name" ?
    llm = ChatOpenAI(temperature=temperature, model_name=chat_model)  # type: ignore
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
    )
    return Agent(executor=agent)
