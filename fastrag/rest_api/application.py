# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/application.py

import argparse

import uvicorn

import fastrag
from fastrag.rest_api.conversation_pipeline_creator import get_conversation_pipeline
from fastrag.rest_api.utils import get_app

config = None
app = get_app()


def get_qa_app(args):
    return fastrag.load_pipeline(args.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app_type",
        type=str,
        required=True,
        choices=["qa", "conversation"],
        help="""
        Type of api to deploy.
        - qa: A pipeline for performing Open Domain Question Answering.
        - conversation: A conversational chat model with retrieval.
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="""
        path to pipeline configuration (yaml).
        If app_type == qa: The config file is the standard Haystack YAML pipeline file.
        If app_type == conversation: The YAML config file is formatted as follows:
        chat_model:
            model_kwargs: {}
            model_name_or_path: liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
            use_gpu: true
        summary_model:
            model_kwargs: {}
            model_name_or_path: meta-llama/Llama-2-7b-chat-hf
            use_gpu: true
        doc_pipeline_file: PATH_TO_DOC_PIPELINE_DILE
        image_pipeline_file: PATH_TO_IMAGE_PIPELINE_DILE
        """,
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="host IP to use for service API calls"
    )
    parser.add_argument("--port", type=int, default=8000, help="service port number to use")
    args = parser.parse_args()

    if args.app_type == "qa":
        app.pipeline = get_qa_app(args)
    elif args.app_type == "conversation":
        app.pipeline = get_conversation_pipeline(args)

    uvicorn.run(app, host=args.host, port=args.port)
