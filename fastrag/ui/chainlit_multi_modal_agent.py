import argparse
import base64
import json
import os
import uuid
from io import BytesIO

import chainlit as cl
from chainlit_agent_fastrag_callback import HaystackAgentCallbackHandler

from fastrag.agents.create_agent import get_agent_conversation_pipeline

# Path to pipeline configuration (yaml).
# An example for a YAML config file is formatted as follows:
# chat_model:
#   generator_kwargs:
#       model: llava-hf/llava-1.5-7b-hf
#       task: "image-to-text"
#       generation_kwargs:
#         max_new_tokens: 100
#       huggingface_pipeline_kwargs:
#         torch_dtype: torch.bfloat16
#   generator_class: fastrag.generators.llava.LlavaHFGenerator
# tools:
#   - type: doc_with_image
#     path: "config/empty_retrieval_pipeline.yaml"
#     index_path: "config/empty_index_pipeline.yaml"
#     params:
#       name: "docRetriever"
#       description: "useful for when you need to retrieve textual documents to answer questions."
#       output_variable: "documents"
config = os.environ.get("CONFIG", "config/visual_chat_agent.yaml")
# delete_on_end_chat = bool(int(os.environ.get("DELETE_ON_END_CHAT", "0")))

args = argparse.Namespace(app_type="conversation", config=config)

agent = get_agent_conversation_pipeline(args)

HaystackAgentCallbackHandler(agent)


@cl.on_chat_end
def chat_end():
    # clear memory
    agent.memory.clear()


def add_images_to_message(additional_params):
    image_elements = []
    last_image = [additional_params["images"][-1]]
    for image_base64_index, image_base64 in enumerate(last_image):
        image_uuid = str(uuid.uuid4())
        bytes_io = BytesIO(base64.b64decode(image_base64))
        image_path = f"imageToSave_{image_base64_index}_{image_uuid}.png"
        with open(image_path, "wb") as fh:
            fh.write(bytes_io.read())

        image_elements.append(cl.Image(name=image_path, display="inline", path=image_path))
    return image_elements


def remove_images(image_elements):
    for e in image_elements:
        os.remove(e.path)


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
        for tool_name in agent.tm.tools:
            tool = agent.tm.tools[tool_name]
            if hasattr(tool, "upload_data_to_pipeline"):
                _ = await cl.Message(
                    author="Agent", content=f"Uploading into {tool.name}..."
                ).send()
                tool.upload_data_to_pipeline(params)

    agent_result = await cl.make_async(agent.run)(message.content)
    answer = agent_result["answers"][0].answer

    # display retrieved image, if exists
    additional_params = agent.memory.get_additional_params()
    if "images" in additional_params and len(additional_params["images"]) > 0:
        image_elements = add_images_to_message(additional_params)

        _ = await cl.Message(
            content="Here are the images I have found.", elements=image_elements
        ).send()
        remove_images(image_elements)

    # Send the agent answer
    _ = await cl.Message(author="Agent", content=answer).send()
