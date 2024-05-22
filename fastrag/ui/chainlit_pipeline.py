import argparse
import base64
import json
import os

import chainlit as cl
from chainlit.sync import run_sync
from events import Events
from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator

from fastrag.agents.base import AgentTokenStreamingHandler, HFTokenStreamingHandler
from fastrag.agents.create_agent import get_basic_conversation_pipeline

config = os.environ.get("CONFIG", "config/rag_pipeline_chat.yaml")

args = argparse.Namespace(app_type="conversation", config=config)

generator, tools_objects = get_basic_conversation_pipeline(args)
callback_manager = Events()

stream_handler = AgentTokenStreamingHandler(callback_manager)
generator.generation_kwargs["streamer"] = HFTokenStreamingHandler(
    generator.pipeline.tokenizer, stream_handler
)

memory = [
    {
        "role": "system",
        "content": """You are a helpful assistant.
Your answers must be short and to the point.
""",
    }
]


@cl.on_chat_end
def chat_end():
    global memory
    # clear memory
    memory = []


@cl.on_message
async def main(message: cl.Message):
    global current_settings

    def parse_element(element, params):
        # insert the image into the params
        if "image" in element.mime:
            img_str = base64.b64encode(element.content).decode("utf-8")
            if "images" not in params:
                params["images"] = []
            params["images"].append(img_str)

        # insert the text into the params
        if "text" in element.mime:
            file_text = element.content.decode("utf-8")
            if "file_texts" not in params:
                params["file_texts"] = []
            params["file_texts"].append(file_text.split("\n"))

        if "json" in element.mime:
            if "data_rows" not in params:
                params["data_rows"] = []

            data_rows = json.load(open(element.path, "r"))
            params["data_rows"].append(data_rows)

    # params for the agent
    params = {}

    if len(message.elements) > 0:
        # parse the input elements, such as images, text, etc.
        for el in message.elements:
            parse_element(el, params)

        # Upload text and images, when appropriate
        for tool in tools_objects:
            if hasattr(tool, "upload_data_to_pipeline"):
                _ = await cl.Message(
                    author="Agent", content=f"Uploading into {tool.name}..."
                ).send()
                tool.upload_data_to_pipeline(params)

    user_query = message.content
    user_input = user_query
    for tool in tools_objects:
        tool_result = tool.run(user_query)
        user_input += tool_result
        _ = await cl.Message(author=tool.name, content=tool_result).send()

    memory.append({"role": "user", "content": user_input})

    prompt = generator.pipeline.tokenizer.apply_chat_template(
        memory, tokenize=False, add_generation_prompt=True
    )

    message = cl.Message(author="Agent", content="")

    def stream_to_message(token):
        run_sync(message.stream_token(token))

    callback_manager.on_new_token = stream_to_message

    result = await cl.make_async(generator.run)(prompt)

    answer = result["replies"][0]

    memory.append({"role": "assistant", "content": answer})
