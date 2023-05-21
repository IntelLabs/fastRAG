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
# See: https://github.com/deepset-ai/haystack/blob/main/ui/utils.py

import logging
import os
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
STATUS = "status"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"
READER_REQUEST = "reader-query"


def haystack_is_ready():
    """
    Used to show the "Haystack is loading..." message
    """
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        if requests.get(url).status_code < 400:
            return True
    except Exception as e:
        logging.exception(e)
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return False


def query(
    query,
    filters={},
    top_k_reader=1,
    top_k_retriever=20,
    top_k_reranker=20,
    diff_steps=None,
    full_pipeline=True,
    pipeline_params_dict=None,
    debug=False,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    pipeline_params_dict is a nested dictionary in general. Each of its keys represents pipeline's component
    and the corresponding key's value is internal dictionary with this component's parameters
    pipeline_params_dict can hold also a key:value pair where value is not dictionary. In this case, this value
    is relevant to all components i.e.
    {"common_param": 42, "Reader": {"max_length": max_summary_length}, "Reranker": {"dist": "cosine"}}
    """
    if full_pipeline:
        url = f"{API_ENDPOINT}/{DOC_REQUEST}"
        params = {
            "filters": filters,
            "Retriever": {"top_k": top_k_retriever},
            "Reranker": {"top_k": top_k_reranker},
            "Reader": {"top_k": top_k_reader},
        }
        if diff_steps is not None:
            params["Image_gen"] = {"num_inference_steps": diff_steps}

        if pipeline_params_dict:
            update_params(params, pipeline_params_dict)
        req = {"query": query, "params": params, "debug": debug}
    else:  # reader only
        url = f"{API_ENDPOINT}/{READER_REQUEST}"
        params = {
            "Reader": {
                "top_k": top_k_reader,
                "documents": query,
            },  # for reader only flow the query holds
        }  # the text of the document to be summarized
        update_params(params, pipeline_params_dict)
        req = {"query": "summarize", "params": params}

    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["answers"]
    for answer in answers:
        if type(answer) == list:
            answer = answer[0]

        if answer.get("answer", None):
            results.append(
                {
                    "answer": answer.get("answer", None),
                    "document": response["documents"],
                    "_raw": answer,
                }
            )
        else:
            results.append(
                {
                    "context": None,
                    "answer": None,
                    "document": None,
                    "relevance": round(answer["score"] * 100, 2),
                    "_raw": answer,
                }
            )

    images = []
    if response.get("images"):
        for from_image in response["images"]:
            image_outputs = response["images"][from_image]
            for image_output in image_outputs:
                images.append(
                    {"text": image_output["merged_text"], "image_content": image_output["image"]}
                )
    relations = response.get("relations", [])

    return results, response, images, relations


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document, document) -> None:
    """
    Send a feedback (label) to the REST API
    """
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "query": query,
        "document": document,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
        "origin": "user-feedback",
        "answer": answer_obj,
    }
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400:
        raise ValueError(
            f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}"
        )


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response


def get_backlink(result) -> Tuple[Optional[str], Optional[str]]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None


def update_params(params_dict, pipeline_params_dict):
    for k in pipeline_params_dict.keys():
        if k in params_dict.keys():
            params_dict[k].update(pipeline_params_dict[k])
        else:
            params_dict[k] = pipeline_params_dict[k]


def display_runtime_plot(raw_json):
    runtimes = raw_json.get("timings")
    total_runtime = 0
    lefts = []
    component_runtime = {}
    for r in runtimes:
        lefts.append(total_runtime)
        total_runtime += float(runtimes[r][1])
        component_runtime[r] = round(float(runtimes[r][1]), 2)

    category_colors = plt.colormaps["inferno"](np.linspace(0.15, 0.85, len(component_runtime)))
    fig, ax = plt.subplots(figsize=(6, 0.4))
    ax.axis("off")
    ax.invert_yaxis()
    for i, (r, color) in enumerate(zip(component_runtime, category_colors)):
        rects = ax.barh(
            "Runtime", float(component_runtime[r]), left=lefts[i], height=0.5, label=r, color=color
        )
        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        ax.bar_label(rects, label_type="center", color=text_color)
    ax.legend(
        ncol=len(component_runtime.keys()),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )
    ax.yaxis.set_visible(False)
    ax.set_xlim(0, total_runtime)
    st.pyplot(fig)
