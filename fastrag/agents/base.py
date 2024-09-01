# SPDX-FileCopyrightText: © 2023 Haystack <https://github.com/deepset-ai/haystack.git>
# SPDX-License-Identifier: Apache License 2.0

import logging
import re
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

from events import Events
from haystack import Answer
from transformers import TextStreamer

from fastrag.agents.agent_step import AgentStep
from fastrag.agents.memory.conversation_memory import ConversationMemory
from fastrag.agents.utils import Color, clean_for_prompt, print_text, react_parameter_resolver

logger = logging.getLogger(__name__)


class AgentTokenStreamingHandler:
    DONE_MARKER = "[DONE]"

    def __init__(self, events: Events):
        self.events = events

    def __call__(self, token_received, **kwargs) -> str:
        self.events.on_new_token(token_received, **kwargs)
        return token_received


class HFTokenStreamingHandler(TextStreamer):
    def __init__(self, tokenizer, stream_handler):
        super().__init__(tokenizer=tokenizer, skip_prompt=True)
        self.token_handler = stream_handler

    def on_finalized_text(self, token: str, stream_end: bool = False):
        token_to_send = token + "\n" if stream_end else token
        self.token_handler(token_received=token_to_send, **{})


class Tool:
    """
    Agent uses tools to find the best answer. A tool is a pipeline or a node. When you add a tool to an Agent, the Agent
    can invoke the underlying pipeline or node to answer questions.

    You must provide a name and a description for each tool. The name should be short and should indicate what the tool
    can do. The description should explain what the tool is useful for. The Agent uses the description to decide when
    to use a tool, so the wording you use is important.

    :param name: The name of the tool. The Agent uses this name to refer to the tool in the text the Agent generates.
        The name should be short, ideally one token, and a good description of what the tool can do, for example:
        "Calculator" or "Search". Use only letters (a-z, A-Z), digits (0-9) and underscores (_)."
    :param description: A description of what the tool is useful for. The Agent uses this description to decide
        when to use which tool. For example, you can describe a tool for calculations by "useful for when you need to

        answer questions about math".
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        logging_color: Color = Color.YELLOW,
    ):
        if re.search(r"\W", name):
            raise ValueError(
                f"Invalid name supplied for tool: '{name}'. Use only letters (a-z, A-Z), digits (0-9) and "
                f"underscores (_)."
            )
        self.name = name
        self.description = description
        self.logging_color = logging_color

    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        raise NotImplementedError()


class ToolsManager:
    """
    The ToolsManager manages tools for an Agent.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        tool_pattern: str = r"Tool:\s*(\w+).*\s*Tool Input:\s*(?:\"([\s\S]*?)\"|((?:.|\n)*))\s*",
    ):
        """
        :param tools: A list of tools to add to the ToolManager. Each tool must have a unique name.
        :param tool_pattern: A regular expression pattern that matches the text that the Agent generates to invoke
            a tool.
        """
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools} if tools else {}
        self.tool_pattern = tool_pattern
        self.callback_manager = Events(("on_tool_start", "on_tool_finish", "on_tool_error"))
        self.tools_query_history = {k: dict() for k in self._tools}

    def clear_tool_history(self):
        self.tools_query_history = {k: dict() for k in self._tools}

    @property
    def tools(self):
        return self._tools

    def get_tool_names(self) -> str:
        """
        Returns a string with the names of all registered tools.
        """
        return ", ".join(self.tools.keys())

    def get_tools(self) -> List[Tool]:
        """
        Returns a list of all registered tool instances.
        """
        return list(self.tools.values())

    def get_tool_names_with_descriptions(self) -> str:
        """
        Returns a string with the names and descriptions of all registered tools.
        """
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools.values()])

    def run_tool(self, llm_response: str, params: Optional[Dict[str, Any]] = None) -> str:
        tool_result: str = ""
        additional_params = None
        if self.tools:
            tool_name, tool_input = self.extract_tool_details(llm_response)
            if tool_name and tool_input:
                if tool_name not in self.tools:
                    return "The Tool I have specified is not valid, I will re-phrase the Tool name and try again."

                # check if tool was already used:
                if tool_input in self.tools_query_history[tool_name]:
                    # return """I have already used this Tool with this Tool Input, I will now use a DIFFERENT Tool and a different Tool Input."""
                    return """I have already used this Tool with this Tool Input. I will use the information I already have to respond."""
                else:
                    self.tools_query_history[tool_name][tool_input] = True

                tool: Tool = self.tools[tool_name]
                try:
                    self.callback_manager.on_tool_start(tool_input, tool=tool)
                    tool_result, additional_params = tool.run(tool_input, params)
                    self.callback_manager.on_tool_finish(
                        tool_result,
                        observation_prefix="Observation: ",
                        llm_prefix="Thought: ",
                        color=tool.logging_color,
                        tool_name=tool.name,
                        tool_input=tool_input,
                    )
                except Exception as e:
                    self.callback_manager.on_tool_error(e, tool=self.tools[tool_name])
                    raise e
        return tool_result, additional_params

    def extract_tool_details(self, llm_response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the tool name and the tool input from the PromptNode response.
        :param llm_response: The PromptNode response.
        :return: A tuple containing the tool name and the tool input.
        """
        tool_match = re.search(self.tool_pattern, llm_response)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_input = tool_match.group(2) or tool_match.group(3)
            return tool_name.strip('" []\n').strip(), tool_input.strip('" \n')
        return None, None


class Agent:
    """
    An Agent answers queries using the tools you give to it. The tools are pipelines or nodes. The Agent uses a large
    language model (LLM) through the PromptNode you initialize it with. To answer a query, the Agent follows this
    sequence:

    1. It generates a thought based on the query.
    2. It decides which tool to use.
    3. It generates the input for the tool.
    4. Based on the output it gets from the tool, the Agent can either stop if it now knows the answer or repeat the
    process of 1) generate thought, 2) choose tool, 3) generate input.

    Agents are useful for questions containing multiple sub questions that can be answered step-by-step (Multi-hop QA)
    using multiple pipelines and nodes as tools.
    """

    def __init__(
        self,
        generator: Any,
        prompt_template: Optional[Union[str, dict]] = None,
        tools_manager: Optional[ToolsManager] = None,
        memory: Optional[ConversationMemory] = None,
        prompt_parameters_resolver: Optional[Callable] = None,
        max_steps: int = 8,
        final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
        streaming: bool = True,
    ):
        """
         Creates an Agent instance.

        :param generator: The Generator that the Agent uses to decide which tool to use and what input to provide to
        it in each iteration.
        :param prompt_template: A new PromptTemplate or the name of an existing PromptTemplate for the PromptNode. It's
        used for generating thoughts and choosing tools to answer queries step-by-step. If it's not set, the PromptNode's
        default template is used and if it's not set either, the Agent's default `zero-shot-react` template is used.
        :param tools_manager: A ToolsManager instance that the Agent uses to run tools. Each tool must have a unique name.
        You can also add tools with `add_tool()` before running the Agent.
        :param memory: A Memory instance that the Agent uses to store information between iterations.
        :param prompt_parameters_resolver: A callable that takes query, agent, and agent_step as parameters and returns
        a dictionary of parameters to pass to the prompt_template. The default is a callable that returns a dictionary
        of keys and values needed for the React agent prompt template.
        :param max_steps: The number of times the Agent can run a tool +1 to let it infer it knows the final answer.
            Set it to at least 2, so that the Agent can run one a tool once and then infer it knows the final answer.
            The default is 8.
        :param final_answer_pattern: A regular expression to extract the final answer from the text the Agent generated.
        :param streaming: Whether to use streaming or not. If True, the Agent will stream response tokens from the LLM.
        If False, the Agent will wait for the LLM to finish generating the response and then process it. The default is
        True.
        """
        self.max_steps = max_steps
        self.tm = tools_manager or ToolsManager()
        self.memory = memory
        self.callback_manager = Events(
            (
                "on_agent_start",
                "on_agent_step",
                "on_agent_finish",
                "on_agent_final_answer",
                "on_new_token",
            )
        )
        self.generator = generator
        self.set_agent_streamer()

        self.prompt_template = prompt_template
        self.prompt_parameters_resolver = (
            prompt_parameters_resolver if prompt_parameters_resolver else react_parameter_resolver
        )
        self.final_answer_pattern = final_answer_pattern
        self.add_default_logging_callbacks(streaming=streaming)

    def set_agent_streamer(self):
        tokenizer = self.generator.generation_kwargs["streamer"].tokenizer
        stream_handler = AgentTokenStreamingHandler(self.callback_manager)
        self.generator.generation_kwargs["streamer"] = HFTokenStreamingHandler(
            tokenizer, stream_handler
        )

    def add_default_logging_callbacks(
        self, agent_color: Color = Color.GREEN, streaming: bool = False
    ) -> None:
        def on_tool_finish(
            tool_output: str,
            color: Optional[Color] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            print_text(observation_prefix)  # type: ignore
            print_text(tool_output, color=color)
            print_text(f"\n{llm_prefix}")

        def on_agent_start(**kwargs: Any) -> None:
            agent_name = kwargs.pop("name", "react")
            print_text(f"\nAgent {agent_name} started with {kwargs}\n")

        def on_agent_final_answer(final_answer: Dict[str, Any], **kwargs: Any) -> None:
            pass

        self.tm.callback_manager.on_tool_finish += on_tool_finish
        self.callback_manager.on_agent_start += on_agent_start
        self.callback_manager.on_agent_final_answer += on_agent_final_answer

        if streaming:
            self.callback_manager.on_new_token += lambda token, **kwargs: print_text(
                token, color=agent_color
            )
        else:
            self.callback_manager.on_agent_step += lambda agent_step: print_text(
                agent_step.generator_node_response, end="\n", color=agent_color
            )

    def add_tool(self, tool: Tool):
        """
        Add a tool to the Agent. This also updates the PromptTemplate for the Agent's PromptNode with the tool name.

        :param tool: The tool to add to the Agent. Any previously added tool with the same name will be overwritten.
        """
        if tool.name in self.tm.tools:
            logger.warning(
                "The agent already has a tool named '%s'. The new tool will overwrite the existing one.",
                tool.name,
            )
        self.tm.tools[tool.name] = tool

    def has_tool(self, tool_name: str) -> bool:
        """
        Check whether the Agent has a tool with the name you provide.

        :param tool_name: The name of the tool for which you want to check whether the Agent has it.
        """
        return tool_name in self.tm.tools

    def run(
        self, query: str, max_steps: Optional[int] = None, params: Optional[dict] = None
    ) -> Dict[str, Union[str, List[Answer]]]:
        """
        Runs the Agent given a query and optional parameters to pass on to the tools used. The result is in the
        same format as a pipeline's result: a dictionary with a key `answers` containing a list of answers.

        :param query: The search query
        :param max_steps: The number of times the Agent can run a tool +1 to infer it knows the final answer.
            If you want to set it, make it at least 2 so that the Agent can run a tool once and then infer it knows the
            final answer.
        :param params: A dictionary of parameters you want to pass to the tools that are pipelines.
                       To pass a parameter to all nodes in those pipelines, use the format: `{"top_k": 10}`.
                       To pass a parameter to targeted nodes in those pipelines, use the format:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}`.
                        You can only pass parameters to tools that are pipelines, but not nodes.
        """
        self.callback_manager.on_agent_start(name="Agent", query=query, params=params)
        agent_step = self.create_agent_step(max_steps)
        first_call = True
        try:
            while not agent_step.is_last():
                agent_step = self._step(query, agent_step, params, first_call)
                first_call = False
        finally:
            self.callback_manager.on_agent_finish(agent_step)
        final_answer = agent_step.final_answer(query=query)
        self.callback_manager.on_agent_final_answer(final_answer)
        return final_answer

    def _step(
        self, query: str, current_step: AgentStep, params: Optional[dict] = None, first_call=False
    ):
        # plan next step using the LLM
        generator_node_response = self._plan(query, current_step, first_call)

        # from the LLM response, create the next step
        next_step = current_step.create_next_step(generator_node_response)
        self.callback_manager.on_agent_step(next_step)

        # run the tool selected by the LLM
        tool_result = (
            self.tm.run_tool(next_step.generator_node_response, params)
            if not next_step.is_last()
            else None
        )

        if tool_result and len(tool_result) == 2:
            observation, additional_params = tool_result
        else:
            observation = tool_result
            additional_params = None

        # save the input, output and observation to memory (if memory is enabled)
        response_to_save = (
            generator_node_response
            if isinstance(generator_node_response, str)
            else generator_node_response[0]
        )
        memory_data = dict(input=query, output=response_to_save, observation=observation)
        if additional_params:
            memory_data["additional_params"] = additional_params

        self.memory.save(data=memory_data, first_call=first_call)

        # update the next step with the observation
        next_step.completed(observation)
        return next_step

    def apply_params_to_template(self, template_params, first_call):
        roles = template_params.pop("memory")
        if isinstance(self.prompt_template, dict):
            # separate system and user/assistant roles
            all_roles = self.prompt_template["system"] + roles
            if first_call:
                all_roles += self.prompt_template["chat"]

                prompt_template = self.generator.pipeline.tokenizer.apply_chat_template(
                    all_roles, tokenize=False, add_generation_prompt=True
                )
            else:
                new_assistant_text = all_roles[-1]["content"]
                prompt_template = (
                    self.generator.pipeline.tokenizer.apply_chat_template(
                        all_roles[:-1], tokenize=False, add_generation_prompt=True
                    )
                    + new_assistant_text
                )

            prompt = prompt_template.format(**template_params)
        else:
            template_params["memory"] = self.generator.pipeline.tokenizer.apply_chat_template(
                roles, tokenize=False, add_generation_prompt=True
            )
            prompt = self.prompt_template.format(**template_params)

        # prompt += "Thought: "
        return prompt

    def _plan(self, query, current_step, first_call):
        # first resolve prompt template params
        template_params = self.prompt_parameters_resolver(
            query=query, agent=self, agent_step=current_step
        )
        additional_params = self.memory.get_additional_params()

        # invoke via prompt node

        prompt = self.apply_params_to_template(template_params, first_call)
        generator_node_response = self.generator.run(prompt=prompt, **additional_params)
        generator_node_response = [clean_for_prompt(r) for r in generator_node_response["replies"]]

        return generator_node_response

    def create_agent_step(self, max_steps: Optional[int] = None) -> AgentStep:
        """
        Create an AgentStep object. Override this method to customize the AgentStep class used by the Agent.
        """
        return AgentStep(
            max_steps=max_steps or self.max_steps, final_answer_pattern=self.final_answer_pattern
        )
