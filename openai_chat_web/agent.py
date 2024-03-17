from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from openai_chat_web import model, tool
from openai_chat_web.history import History, Message


class Agent:
    __executor: AgentExecutor

    def __init__(self, executor: AgentExecutor):
        self.__executor = executor

    def send(self, messages: History) -> Message:
        chat_history = messages.into_bases()
        input_message = chat_history.pop().content
        result = self.__executor.invoke(
            {
                "chat_history": chat_history,
                "input": input_message,
            }
        )
        return Message(role="ai", content=result["output"])


def new(
    chat_model: str = "gpt-3.5-turbo", temperature: float = 0.7, language: str = "English", verbose: bool = False
) -> Agent:
    """Return a new `Agent`."""
    tools = [
        tool.wikipedia(),
        tool.tavily(),
        tool.duckduckgo(),
    ]
    llm = model.chat(model=chat_model, temperature=temperature)

    ai_message = f"""You are a helpful assistant. \
    Provide the best answers to user questions in {language}. \
    Use tools only when necessary."""
    history_placeholder = MessagesPlaceholder(variable_name="chat_history", optional=True)
    user_message = "{input}"
    scratchpad_placeholder = MessagesPlaceholder(variable_name="agent_scratchpad")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", ai_message),
            history_placeholder,
            ("user", user_message),
            scratchpad_placeholder,
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)  # type: ignore
    return Agent(executor=executor)
