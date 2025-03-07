"""Entry point of CLI."""

import sys
import traceback

import pkommand

from openai_chat_web import agent, history


class NoInputError(Exception):
    """No input to chat."""


def chat(
    role_separator: str = ">\n",
    message_separator: str = "\n---\n",
    chat_model: str = "gpt-4o",
    temperature: float = 0.7,
    language: str = "English",
    verbose: bool = False,
):
    r"""
    Start chat.

    Receive messages from stdin in the following format:

    ROLE>
    CONTENT

    Pass multiple messages:

    ROLE>
    CONTENT
    ---
    ROLE>
    CONTENT

    '>\\n' is the default role_separator.
    '\\n---\\n' is the default message_separator.

    If role is omitted, role is user_role, e.g.

    CONTENT

    equals

    ROLE>
    CONTENT

    user_role is Human role name, default is user.
    ai_role is AI role name, default is ai.

    Chat replies are written to stdout, e.g.

    $ echo 'Hello!' | openai_chat_web chat
    user>
    Hello!
    ---
    ai>
    Hello! How can I assist you today?

    Default temperature is 0.7.

    If verbose is true, display verbose output.
    """
    memory = history.parse(
        buffer=sys.stdin.read(),
        role_separator=role_separator,
        message_separator=message_separator,
    )
    if not memory:
        raise NoInputError()

    api = agent.new(
        chat_model=chat_model,
        temperature=temperature,
        language=language,
        verbose=verbose,
    )
    response = api.send(memory)
    memory.append(response)

    print(memory.into_str(message_sep=message_separator, role_sep=role_separator))


def main() -> int:
    """Entry point of CLI."""
    try:
        wrapper = pkommand.Wrapper(pkommand.Parser("openai-chat-web"))
        wrapper.add(chat)
        wrapper.run()
        return 0
    except Exception as e:
        traceback.print_exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
