import torch
import yaml
from transformers import AutoTokenizer, StoppingCriteriaList

from fastrag.agents.base import Agent, ToolsManager
from fastrag.agents.memory.conversation_memory import ConversationMemory
from fastrag.agents.tool_handlers import INDEX_HANDLER_FACTORY, QUERY_HANDLER_FACTORY
from fastrag.agents.tools.tools import TOOLS_FACTORY
from fastrag.generators.stopping_criteria.stop_words import StopWordsByTextCriteria

AGENT_SYSTEM_ROLES = [
    {
        "role": "system",
        "content": """You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_names_with_descriptions}

## Output Format

Please ALWAYS start with "Thought:".
If you need to use a tool to get more information to answer the question, use the following format:

```
Thought: I need to use a tool to help me answer the question.
Tool: [tool name if using a tool].
Tool Input: [the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world"}})].
Observation: [tool response]
```

Please use a valid JSON format for the Action Input.
Please ALWAYS start with "Thought:".

You should keep repeating the above format till you have enough information to answer the question without using any more tools.
If you have enough information to answer the question without using any more tools, you MUST finish with "Final Answer: and respond in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Final Answer: [your answer here ]
```

If you have enough information to answer the question without using any more tools, you MUST finish with "Final Answer:".

""",
    },
]

AGENT_CONVERSATION_BASE_ROLES = [
    {"role": "user", "content": " {query} "},
    {
        "role": "assistant",
        "content": """
{transcript}
""",
    },
]

AGENT_ROLES = {"system": AGENT_SYSTEM_ROLES, "chat": AGENT_CONVERSATION_BASE_ROLES}


def replace_class_names_with_types(entry, k=None, parent=None):
    """
    Replaces the class names specified in the YAML config file with the actual types objects.
    This needs to be done in order to properly load the invocation layers and torch types.
    """
    if type(entry) == list:
        for i, item in enumerate(entry):
            replace_class_names_with_types(item, i, entry)

    if type(entry) == dict:
        for key, value in entry.items():
            replace_class_names_with_types(value, key, entry)

    # automatically replace torch_dtype strings with types
    if k == "torch_dtype":
        parent[k] = getattr(torch, entry.split(".")[1])


def get_generator(chat_model_config):
    class_path_parts = chat_model_config["generator_class"].split(".")
    current_module = __import__(class_path_parts[0])
    for part in class_path_parts[1:]:
        current_module = getattr(current_module, part)
    generator_class = current_module

    tokenizer = AutoTokenizer.from_pretrained(chat_model_config["generator_kwargs"]["model"])
    # stop_word_list = ["Observation:"]
    stop_word_list = ["Observation:", "<|eot_id|>"]
    sw = StopWordsByTextCriteria(tokenizer=tokenizer, stop_words=stop_word_list, device="cpu")

    if "generation_kwargs" in chat_model_config["generator_kwargs"]:
        chat_model_config["generator_kwargs"]["generation_kwargs"] = {}

    chat_model_config["generator_kwargs"]["generation_kwargs"][
        "stopping_criteria"
    ] = StoppingCriteriaList([sw])

    replace_class_names_with_types(chat_model_config)

    generator = generator_class(**chat_model_config["generator_kwargs"])

    generator.warm_up()
    return generator


def get_agent_conversation_pipeline(args):
    conversation_config = yaml.safe_load(open(args.config, "r"))

    chat_model_config = conversation_config["chat_model"]

    generator = get_generator(chat_model_config)

    tools_objects = []
    if "tools" in conversation_config:
        tools = conversation_config["tools"]
        for tool_config in tools:
            tool_type = tool_config["type"]
            tool_type_class = TOOLS_FACTORY[tool_type]

            tool_params = tool_config["params"]

            # Create QueryHandler
            query_handler_type = QUERY_HANDLER_FACTORY[tool_config["query_handler"]["type"]]
            query_handler = query_handler_type(**tool_config["query_handler"]["params"])

            store = query_handler.get_store()

            # Create IndexHandler
            index_handler_type = INDEX_HANDLER_FACTORY[tool_config["index_handler"]["type"]]

            tool_obj = tool_type_class(
                query_handler=query_handler,
                index_handler=index_handler_type(
                    document_store=store,
                    **tool_config["index_handler"]["params"],
                ),
                **tool_params,
            )

            tools_objects.append(tool_obj)

    tools_manager = ToolsManager(tools_objects)

    conversational_agent = Agent(
        generator,
        prompt_template=AGENT_ROLES,
        memory=ConversationMemory(generator=generator),
        tools_manager=tools_manager,
    )

    return conversational_agent
