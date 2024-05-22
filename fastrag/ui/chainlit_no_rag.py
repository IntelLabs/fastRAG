import os

import chainlit as cl
import yaml
from chainlit.sync import run_sync
from events import Events
from transformers import AutoTokenizer, pipeline

from fastrag.agents.base import AgentTokenStreamingHandler, HFTokenStreamingHandler

config_path = os.environ.get("CONFIG", "config/regular_chat.yaml")
model_kwargs = yaml.safe_load(open(config_path, "r"))

callback_manager = Events()

model_name_or_path = model_kwargs["model"]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
stream_handler = AgentTokenStreamingHandler(callback_manager)
streamer = HFTokenStreamingHandler(tokenizer, stream_handler)
model_pipeline = pipeline(**model_kwargs, return_full_text=False, streamer=streamer)

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

    memory.append({"role": "user", "content": message.content})

    prompt = model_pipeline.tokenizer.apply_chat_template(
        memory, tokenize=False, add_generation_prompt=True
    )

    message = cl.Message(author="Agent", content="")

    def stream_to_message(token):
        run_sync(message.stream_token(token))

    callback_manager.on_new_token = stream_to_message

    result = await cl.make_async(model_pipeline)(prompt)

    answer = result[0]["generated_text"]

    memory.append({"role": "assistant", "content": answer})
