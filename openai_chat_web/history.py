from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def into_str(self, sep: str = ":") -> str:
        return self.role + sep + self.content


class History(list[Message]):
    def into_str(self, message_sep: str = "\n", role_sep: str = ":") -> str:
        return message_sep.join(x.into_str(role_sep) for x in self)


def parse(
    buffer: str,
    role_separator: str,
    message_separator: str,
    default_role: str,
) -> History:
    """Parse strings as  `History`."""

    def parse_message(message: str) -> Message:
        if role_separator not in message:
            return Message(role=default_role, content=message)

        role, content = message.split(role_separator, maxsplit=1)
        return Message(role=role, content=content)

    messages = [parse_message(x.lstrip().rstrip()) for x in buffer.split(message_separator)]
    return History(messages)
