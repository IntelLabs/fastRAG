# SPDX-FileCopyrightText: © 2023 Haystack <https://github.com/deepset-ai/haystack.git>
# SPDX-License-Identifier: Apache License 2.0

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastrag.agents.agent_step import AgentStep

if TYPE_CHECKING:
    from fastrag.agents import Agent


class Color(Enum):
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\x1b[0m"


def load_text(path):
    return open(path, "r").read()


def clean_for_prompt(text):
    # ensure that {} do not appear in the prompt, to avoid format error.
    return text.replace("{", "{{").replace("}", "}}")


def print_text(text: str, end="", color: Optional[Color] = None) -> None:
    """
    Print text with optional color.
    :param text: Text to print.
    :param end: End character to use (defaults to "").
    :param color: Color to print text in (defaults to None).
    """
    if color:
        print(f"{color.value}{text}{Color.RESET.value}", end=end, flush=True)
    else:
        print(text, end=end, flush=True)


def react_parameter_resolver(
    query: str, agent: "Agent", agent_step: AgentStep, **kwargs
) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct-based conversational agent that returns the query, the tool names, the tool names
    with descriptions, the history of the conversation, and the transcript (internal monologue).
    """
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }
