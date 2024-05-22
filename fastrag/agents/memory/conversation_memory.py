# SPDX-FileCopyrightText: © 2023 Haystack <https://github.com/deepset-ai/haystack.git>
# SPDX-License-Identifier: Apache License 2.0

import collections
from typing import Any, Dict, List, Optional, OrderedDict


class ConversationMemory:
    """
    A memory class that stores conversation history.
    """

    def __init__(self, input_key: str = "input", output_key: str = "output", generator: Any = None):
        """
        Initialize ConversationMemory with input and output keys.

        :param input_key: The key to use for storing user input.
        :param output_key: The key to use for storing model output.
        """
        self.list: List[OrderedDict] = []
        self.input_key = input_key
        self.output_key = output_key
        self.generator = generator
        self.generator_modifies_user_text = hasattr(self.generator, "get_user_text")

    def get_user_text(self, chat_snippet):
        if self.generator_modifies_user_text:
            user_text = self.generator.get_user_text(chat_snippet)
        else:
            user_text = chat_snippet["Human"]
        return {"role": "user", "content": user_text}

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
        """
        Load conversation history as a formatted string.

        :param keys: Optional list of keys (ignored in this implementation).
        :param kwargs: Optional keyword arguments
            - window_size: integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history.
        """
        window_size = kwargs.get("window_size", None)

        if window_size is not None:
            chat_list = self.list[-window_size:]  # pylint: disable=invalid-unary-operand-type
        else:
            chat_list = self.list

        roles = []
        for chat_snippet in chat_list:
            user_role = self.get_user_text(chat_snippet)

            assistant_text = f"{chat_snippet['AI']}\n"
            # assistant_text = f"Assistant: {chat_snippet['AI']}\n"
            if "observation" in chat_snippet:
                assistant_text += f"Observation: {chat_snippet['observation']}\n"
            assistant_role = {"role": "assistant", "content": assistant_text}

            roles.extend([user_role, assistant_role])

        return roles

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a conversation snippet to memory.

        :param data: A dictionary containing the conversation snippet to save.
        """
        chat_snippet = collections.OrderedDict()
        chat_snippet["Human"] = data[self.input_key]
        chat_snippet["AI"] = data[self.output_key]

        for param_key in ["observation", "additional_params"]:
            if data.get(param_key, None):
                chat_snippet[param_key] = data[param_key]
        self.list.append(chat_snippet)

    def get_additional_params(self):
        params = {}
        for chat_snippet in self.list:
            additional_params = chat_snippet.get("additional_params", {})
            if len(additional_params) > 0:
                for key in params:
                    if key in additional_params:
                        params[key] += additional_params[key]

                for key in additional_params:
                    if key not in params:
                        params[key] = additional_params[key]
        return params

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.list = []
