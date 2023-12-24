from typing import List, Optional

from haystack.agents.memory import ConversationMemory
from haystack.nodes.prompt import PromptTemplate


def llama_2_chat_load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
    """
    Load conversation history as a formatted string.

    :param keys: Optional list of keys (ignored in this implementation).
    :param kwargs: Optional keyword arguments
        - window_size: integer specifying the number of most recent conversation snippets to load.
    :return: A formatted string containing the conversation history.
    """
    chat_transcript = ""
    window_size = kwargs.get("window_size", None)

    if window_size is not None:
        chat_list = self.list[-window_size:]  # pylint: disable=invalid-unary-operand-type
    else:
        chat_list = self.list

    for chat_snippet_index, chat_snippet in enumerate(chat_list):
        if chat_snippet_index == 0:
            user_text = f"{chat_snippet['Human']} [/INST]"
        else:
            user_text = f"<s>[INST] {chat_snippet['Human']} [/INST]"

        if chat_snippet_index == len(chat_list) - 1:
            ai_text = f" {chat_snippet['AI']} </s><s>[INST] "
        else:
            ai_text = f" {chat_snippet['AI']} </s>"

        chat_transcript += user_text  # f"Human: {chat_snippet['Human']}\n"
        chat_transcript += ai_text  # f"AI: {chat_snippet['AI']}\n"
    return chat_transcript


def user_assistant_chat_load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
    """
    Load conversation history as a formatted string.

    :param keys: Optional list of keys (ignored in this implementation).
    :param kwargs: Optional keyword arguments
        - window_size: integer specifying the number of most recent conversation snippets to load.
    :return: A formatted string containing the conversation history.
    """
    chat_transcript = ""
    window_size = kwargs.get("window_size", None)

    if window_size is not None:
        chat_list = self.list[-window_size:]  # pylint: disable=invalid-unary-operand-type
    else:
        chat_list = self.list

    for chat_snippet in chat_list:
        chat_transcript += f"### User: {chat_snippet['Human']}\n"
        chat_transcript += f"### Assistant: {chat_snippet['AI']}\n"
    return chat_transcript


def user_assistant_chat_llava_load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
    """
    Load conversation history as a formatted string.

    :param keys: Optional list of keys (ignored in this implementation).
    :param kwargs: Optional keyword arguments
        - window_size: integer specifying the number of most recent conversation snippets to load.
    :return: A formatted string containing the conversation history.
    """
    chat_transcript = ""
    window_size = kwargs.get("window_size", None)

    if window_size is not None:
        chat_list = self.list[-window_size:]  # pylint: disable=invalid-unary-operand-type
    else:
        chat_list = self.list

    for chat_snippet in chat_list:
        chat_transcript += f"USER: {chat_snippet['Human']}\n"
        chat_transcript += f"ASSISTANT: {chat_snippet['AI']}\n"
    return chat_transcript


# General Chat Prompt Template
def get_default_template():
    return PromptTemplate(
        "The following is a conversation between a human and an AI. Do not generate the user response to your output. {new_line}{memory}{new_line}Human: {query}{new_line}AI:"
    )


# Chat Prompt Template for Llama models
def get_llama_2_template():
    # in case we wish to use a llama-2 style chat, we override the ConversationMemory.load method
    ConversationMemory.load = llama_2_chat_load

    return PromptTemplate(
        """<s>[INST] <<SYS>>
The following is a conversation between a human and an AI. Do not generate the user response to your output.
<</SYS>>

{memory}{query} [/INST] """
    )


# Chat Prompt Template for System - User - Assistant
def get_user_assistant_template():
    # in case we wish to use a llama-2 style chat, we override the ConversationMemory.load method
    ConversationMemory.load = user_assistant_chat_load

    return PromptTemplate(
        """### System:
The following is a conversation between a human and an AI. Do not generate the user response to your output.
{memory}

### User: {query}
### Assistant: """
    )


def get_user_assistant_llava_template():
    # in case we wish to use a llama-2 style chat, we override the ConversationMemory.load method
    ConversationMemory.load = user_assistant_chat_llava_load

    return PromptTemplate(
        """{memory}

USER: {query}
ASSISTANT: """
    )


CHAT_TEMPLATE_MAP = {
    "Llama2": get_llama_2_template,
    "UserAssistant": get_user_assistant_template,
    "UserAssistantLlava": get_user_assistant_llava_template,
}


def get_chat_prompt_template(conversation_config):
    if "chat_template" in conversation_config:
        chat_template_type = conversation_config["chat_template"]  # choose the correct template
        return CHAT_TEMPLATE_MAP[
            chat_template_type
        ]()  # override the chatting behavior and return the template

    return get_default_template()
