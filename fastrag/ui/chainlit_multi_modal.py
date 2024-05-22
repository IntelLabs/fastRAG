import argparse
import base64
import os
from io import BytesIO

import chainlit as cl
from chainlit.input_widget import Select, Slider

from fastrag.rest_api.conversation_pipeline_creator import get_conversation_pipeline

# Path to pipeline configuration (yaml).
# The YAML config file is formatted as follows:
#    chat_model:
#        model_kwargs: {}
#        model_name_or_path: liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
#        use_gpu: true
#    summary_model:
#        model_kwargs: {}
#        model_name_or_path: meta-llama/Llama-2-7b-chat-hf
#        use_gpu: true
#    doc_pipeline_file: PATH_TO_DOC_PIPELINE_DILE
#    image_pipeline_file: PATH_TO_IMAGE_PIPELINE_DILE
config = os.environ.get("CONFIG", "config/visual_chat.yaml")
delete_on_end_chat = bool(int(os.environ.get("DELETE_ON_END_CHAT", "0")))

args = argparse.Namespace(app_type="conversation", config=config)

pipeline = get_conversation_pipeline(args)

cl.HaystackAgentCallbackHandler(pipeline.agent)


MAP_ACTIONS = {
    "Regular Message": dict(use_retrieval=False, use_image_retrieval=False),
    "Do Image Retrieval": dict(use_retrieval=False, use_image_retrieval=True),
    "Do Text Retrieval": dict(use_retrieval=True, use_image_retrieval=False),
    "Do Image + Text Retrieval": dict(use_retrieval=True, use_image_retrieval=True),
}

current_settings = None


@cl.on_chat_start
async def start():
    global current_settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="message_action",
                label="Message Action",
                values=list(MAP_ACTIONS.keys()),
                initial_index=0,
            ),
            Slider(
                id="top_k_retriever",
                label="Documents to Retrieve",
                initial=10,
                min=1,
                max=1000,
                step=1,
            ),
            Slider(
                id="top_k_reranker",
                label="Documents to Re-Rank",
                initial=1,
                min=1,
                max=1000,
                step=1,
            ),
        ]
    ).send()

    current_settings = settings


@cl.on_settings_update
async def setup_agent(settings):
    global current_settings
    current_settings = settings


@cl.on_chat_end
def chat_end():
    pipeline.agent.memory.clear()
    if delete_on_end_chat:
        pipeline.delete_all_data({})


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

    # params for the agent
    params = {
        "inputs": [],
        "ret_pipe": {
            "Retriever": {"top_k": current_settings["top_k_retriever"]},
            "Reranker": {"top_k": current_settings["top_k_reranker"]},
        },
        "debug": True,
    }

    if len(message.elements) > 0:
        # parse the input elements, such as images, text, etc.
        for el in message.elements:
            parse_element(el, params)

        # Upload text and images, when appropriate
        if "file_texts" in params:
            _ = await cl.Message(author="Agent", content="Uploading Text...").send()
            pipeline.upload_docs(params)

        if "images" in params:
            _ = await cl.Message(author="Agent", content="Uploading Images...").send()
            pipeline.upload_images(params)

    agent_result, returned_docs, _, images_to_return = await cl.make_async(pipeline.run)(
        message.content, params, **MAP_ACTIONS[current_settings["message_action"]]
    )
    answer = agent_result["answers"][0].answer

    # Send a message stating that an image has been retrieved
    if len(images_to_return) > 0:
        image_base64 = images_to_return["images"][0]

        bytes_io = BytesIO(base64.b64decode(image_base64))
        with open("imageToSave.png", "wb") as fh:
            fh.write(bytes_io.read())

        image_elements = [cl.Image(name="image1", display="inline", path="imageToSave.png")]

        _ = await cl.Message(
            content="Here is the image I have found.", elements=image_elements
        ).send()

    # Send a message stating that a text document has been retrieved
    if len(returned_docs):
        text_elements = [
            cl.Text(name=f"Document {doc_i+1}", content=doc.content, display="inline")
            for doc_i, doc in enumerate(returned_docs)
        ]

        await cl.Message(
            content="Here are the documents I have found.", elements=text_elements
        ).send()

    # Send the agent answer
    _ = await cl.Message(author="Agent", content=answer).send()
