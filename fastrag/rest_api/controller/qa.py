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
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/controller/search.py

import collections
import json
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, FastAPI
from haystack.nodes import PromptNode, PromptTemplate
from haystack.schema import Answer, Document
from pydantic import BaseConfig

from ..schema import QueryRequest, QueryResponse

BaseConfig.arbitrary_types_allowed = True

from ..utils import get_app

logger = logging.getLogger(__name__)


router = APIRouter()
app: FastAPI = get_app()


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """

    result = _process_request(app.pipeline, request)
    return result


def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()
    params = request.params or {}

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key in params.keys():
        if isinstance(params[key], collections.abc.Mapping) and "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    if "generation_kwargs" in params:
        for n in pipeline.components.values():
            if isinstance(n, PromptNode):
                params.update(
                    {
                        n.name: {
                            "invocation_context": {"generation_kwargs": params["generation_kwargs"]}
                        }
                    }
                )
        del params["generation_kwargs"]

    if "input_prompt" in params:
        new_prompt = PromptTemplate(**params["input_prompt"])

        for n in pipeline.components.values():
            if isinstance(n, PromptNode):
                n.default_prompt_template = new_prompt
        del params["input_prompt"]

    pipeline_components_list = list(pipeline.components.keys())
    for p in list(params.keys()):
        if "filters" == str(p):
            continue
        if str(p) not in pipeline_components_list:
            del params[p]

    result = pipeline.run(query=request.query, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if "documents" not in result:
        result["documents"] = []
    if "answers" not in result:
        result["answers"] = []
    if "results" in result:
        result["answers"] = [Answer(res, "generative") for res in result["results"]]

    logger.info(
        json.dumps(
            {
                "request": request,
                "response": result,
                "time": f"{(time.time() - start_time):.2f}",
            },
            default=str,
        )
    )
    return result


@router.post("/chat", response_model=QueryResponse, response_model_exclude_none=True)
def chat(request: QueryRequest):
    """
    This endpoint receives the user input as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """

    result = _process_request_chat(app.pipeline, request)
    return result


@router.post("/reset_chat", response_model=QueryResponse, response_model_exclude_none=True)
def reset_chat(request: QueryRequest):
    """
    This endpoint receives the user input as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    app.pipeline.agent.memory.clear()
    return {"query": "Chat Reset Complete"}


@router.post("/upload_docs", response_model=QueryResponse, response_model_exclude_none=True)
def upload_docs(request: QueryRequest):
    """
    This endpoint receives the user input as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    app.pipeline.upload_docs(request.params)
    return {"query": "Upload DOCS Complete"}


@router.post("/upload_images", response_model=QueryResponse, response_model_exclude_none=True)
def upload_images(request: QueryRequest):
    """
    This endpoint receives the user input as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    app.pipeline.upload_images(request.params)
    return {"query": "Upload IMAGES Complete"}


@router.post("/delete_all_data", response_model=QueryResponse, response_model_exclude_none=True)
def upload_docs(request: QueryRequest):
    """
    This endpoint receives the user input as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    app.pipeline.delete_all_data(request.params)
    return {"query": "Deleted all data"}


def _process_request_chat(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()
    params = request.params or {}

    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    result, returned_docs, user_prompt, image_data = pipeline.run(
        query=request.query,
        params=params,
        use_retrieval=params["use_retrieval"],
        use_image_retrieval=params["use_image_retrieval"],
    )

    # Ensure answers and documents exist, even if they're empty lists
    if "documents" not in result:
        result["documents"] = []
    if "answers" not in result:
        result["answers"] = []
    if "results" in result:
        result["answers"] = [Answer(res, "generative") for res in result["results"]]

    result["query"] = user_prompt
    result["documents"] = returned_docs
    result["images"] = image_data if "images" in image_data else {}

    logger.info(
        json.dumps(
            {
                "request": request,
                "response": result,
                "time": f"{(time.time() - start_time):.2f}",
            },
            default=str,
        )
    )
    return result


def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(
            f"Request with deprecated filter format ('\"filters\": null'). "
            f"Remove empty filters from params to be compliant with future versions"
        )
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(
                    f"Request with deprecated filter format ('{key}: null'). "
                    f"Remove null values from filters to be compliant with future versions"
                )
                continue

            if not isinstance(values, list):
                logger.warning(
                    f"Request with deprecated filter format ('{key}': {values}). "
                    f"Change to '{key}':[{values}]' to be compliant with future versions"
                )
                values = [values]

            new_filters[key] = values
    return new_filters


@router.post("/reader-query", response_model=QueryResponse, response_model_exclude_none=True)
def reader_query(request: QueryRequest):
    """
    This endpoint receives the query as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    This post request command is intended to a reader only pipeline
    """
    result = _process_request_reader_only(app.pipelines["reader_only"], request)
    return result


def _process_request_reader_only(pipeline, request):
    start_time = time.time()

    request.params["Reader"]["documents"] = [Document(request.params["Reader"]["documents"])]
    params = request.params or {}

    result = pipeline.run(query=request.query, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if "documents" not in result:
        result["documents"] = []
    if "answers" not in result:
        result["answers"] = []

    logger.info(
        json.dumps(
            {
                "request": request,
                "response": result,
                "time": f"{(time.time() - start_time):.2f}",
            },
            default=str,
        )
    )
    return result
