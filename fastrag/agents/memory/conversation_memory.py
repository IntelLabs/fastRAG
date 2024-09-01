# SPDX-FileCopyrightText: © 2023 Haystack <https://github.com/deepset-ai/haystack.git>
# SPDX-License-Identifier: Apache License 2.0

import collections
import logging
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

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
        """
        Load conversation history as a formatted string.

        :param keys: Optional list of keys (ignored in this implementation).
        :param kwargs: Optional keyword arguments
            - window_size: integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history.
        """

        chat_list = self.list

        roles = []
        for chat_snippet in chat_list:
            user_role = {"role": "user", "content": chat_snippet["Human"]}

            assistant_text = f"{chat_snippet['AI']}\n"

            assistant_role = {"role": "assistant", "content": assistant_text}

            roles.extend([user_role, assistant_role])

        return roles

    def save(self, data: Dict[str, Any], first_call=False) -> None:
        """
        Save a conversation snippet to memory.

        :param data: A dictionary containing the conversation snippet to save.
        """
        assistant_text = data[self.output_key]
        if "observation" in data and data["observation"] and data["observation"] != "None":
            assistant_text += f"\n{data['observation']}\n"

        if first_call:
            # in the first call, add a new user - assistant pair
            chat_snippet = collections.OrderedDict()
            chat_snippet["Human"] = data[self.input_key]
            chat_snippet["AI"] = assistant_text

            if data.get("additional_params", None):
                chat_snippet["additional_params"] = data["additional_params"]
            self.list.append(chat_snippet)
        else:
            # edit last assistant message
            last_snippet = self.list[-1]
            last_snippet["AI"] += assistant_text

            if data.get("additional_params", None):
                if last_snippet.get("additional_params", None):
                    for k, v in data["additional_params"].items():
                        if k in last_snippet["additional_params"]:
                            last_snippet["additional_params"][k] += v
                else:
                    last_snippet["additional_params"] = data["additional_params"]

            self.list[-1] = last_snippet

    def get_additional_params(self):
        params = {}
        for chat_snippet in self.list:
            additional_params = chat_snippet.get("additional_params", {})
            if len(additional_params) > 0:
                for key in additional_params:
                    if key not in params:
                        params[key] = []
                    assert isinstance(additional_params[key], list)
                    params[key] += additional_params[key]

        return params

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.list = []
