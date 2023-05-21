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
import time
from typing import Any, Dict

from fastapi import APIRouter, FastAPI
from haystack.nodes import PromptNode, PromptTemplate
from haystack.schema import Answer, Document
from pydantic import BaseConfig

from ..schema import QueryRequest, QueryResponse

BaseConfig.arbitrary_types_allowed = True

import fastrag

from ..utils import get_app

logger = fastrag.utils.init_logger(__name__)

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
        if isinstance(params[key], collections.Mapping) and "filters" in params[key].keys():
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
                template_names = n.get_prompt_template_names()
                if new_prompt.name in template_names:
                    del n.prompt_templates[new_prompt.name]
                n.add_prompt_template(new_prompt)
                n.set_default_prompt_template(new_prompt)
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
