from dataclasses import asdict

import pytest

import openai_chat_web.history as history


@pytest.mark.parametrize(
    "buffer,role_sep,message_sep,default_role,want",
    [
        (
            "R:C",
            ":",
            "$",
            "U",
            history.History(
                [
                    history.Message(role="R", content="C"),
                ]
            ),
        ),
        (
            "C",
            ":",
            "$",
            "U",
            history.History(
                [
                    history.Message(role="U", content="C"),
                ]
            ),
        ),
        (
            "R1:C1$R2:C2",
            ":",
            "$",
            "U",
            history.History(
                [
                    history.Message(role="R1", content="C1"),
                    history.Message(role="R2", content="C2"),
                ]
            ),
        ),
    ],
)
def test_history_parse(buffer: str, role_sep: str, message_sep: str, default_role: str, want: history.History):
    assert [asdict(x) for x in want] == [asdict(x) for x in history.parse(buffer, role_sep, message_sep, default_role)]


@pytest.mark.parametrize(
    "hist,message_sep,role_sep,want",
    [
        (
            history.History([]),
            "\n",
            ":",
            "",
        ),
        (
            history.History(
                [
                    history.Message(role="R", content="C"),
                ]
            ),
            "\n",
            ":",
            "R:C",
        ),
        (
            history.History(
                [
                    history.Message(role="R1", content="C1"),
                    history.Message(role="R2", content="C2"),
                ]
            ),
            "\n",
            ":",
            "R1:C1\nR2:C2",
        ),
    ],
)
def test_history_into_str(hist: history.History, message_sep: str, role_sep: str, want: str):
    assert want == hist.into_str(message_sep, role_sep)
