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
# See: https://github.com/deepset-ai/haystack/blob/main/ui/webapp.py
import logging
import os
from json import JSONDecodeError
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from annotated_text import annotation
from markdown import markdown

from fastrag.ui.kg import get_kg_html
from fastrag.ui.utils import (
    API_ENDPOINT,
    display_runtime_plot,
    haystack_is_ready,
    query,
    upload_doc,
)

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "Who is Barack Obama?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "Paris")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "100"))
DEFAULT_DOCS_FROM_RERANKER = int(os.getenv("DEFAULT_DOCS_FROM_RERANKER", "20"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "1"))
DEFAULT_TASK = os.getenv("DEFAULT_TASK", "qa_pipeline.yaml")
DEFAULT_NUMBER_OF_DIFF_STEPS = int(os.getenv("DEFAULT_NUMBER_OF_DIFF_STEPS", "20"))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", str(Path(__file__).parent / "eval_labels_example.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))
DISABLE_FILE_UPLOAD = True


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():

    st.set_page_config(page_title="fastRAG Demo", page_icon="")

    with open("fastrag/ui/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)
    set_state_if_absent("images", None)
    set_state_if_absent("relations", None)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None
        st.session_state.images = None
        st.session_state.relations = None

    # Title
    st.write("# Q&A Demo")
    st.markdown(
        """
This is a demo of a generative Q&A pipeline, using the fastRAG package.
""",
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("Options")
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=500,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )
    top_k_reranker = st.sidebar.slider(
        "Max. number of documents from reranker",
        min_value=1,
        max_value=30,
        value=DEFAULT_DOCS_FROM_RERANKER,
        step=1,
        on_change=reset_results,
    )
    top_k_reader = st.sidebar.slider(
        "Max. number of answers",
        min_value=1,
        max_value=10,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        on_change=reset_results,
    )
    diff_steps = None
    if DEFAULT_TASK == "qa_diffusion_pipeline.yaml":
        diff_steps = st.sidebar.slider(
            "Number of steps for the diffusion model",
            min_value=5,
            max_value=100,
            value=DEFAULT_NUMBER_OF_DIFF_STEPS,
            step=1,
            on_change=reset_results,
        )
    show_runtime = st.sidebar.checkbox("Show components runtime")
    debug = st.sidebar.checkbox("Show debug info")

    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader(
            "", type=["pdf", "txt", "docx"], accept_multiple_files=True
        )
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")
                if debug:
                    st.subheader("REST API JSON response")
                    st.sidebar.write(raw_json)

    st.sidebar.markdown(
        f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Search bar
    question = st.text_input(
        "", value=st.session_state.question, max_chars=100, on_change=reset_results
    )

    # Run button
    run_pressed = st.button("Run")

    run_query = (
        run_pressed or question != st.session_state.question
    ) and not st.session_state.random_question_requested

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; fastRAG demo is starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is the fastRAG pipeline service running?")
            st.error(f"Using endpoint: {API_ENDPOINT}")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question

        with st.spinner("Searching through documents and generating answers ... \n "):
            try:
                (
                    st.session_state.results,
                    st.session_state.raw_json,
                    st.session_state.images,
                    st.session_state.relations,
                ) = query(
                    question,
                    top_k_retriever=top_k_retriever,
                    top_k_reranker=top_k_reranker,
                    top_k_reader=top_k_reader,
                    diff_steps=diff_steps,
                )
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?"
                )
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                    st.error(f"{e}")
                return

    if show_runtime and st.session_state.results and "timings" in st.session_state.raw_json:
        display_runtime_plot(st.session_state.raw_json)

    if st.session_state.images or st.session_state.results:
        st.write("## Results:")

    if st.session_state.relations:
        source_code = get_kg_html(st.session_state.relations)
        components.html(source_code, height=700, width=700)

    if st.session_state.images:

        image_markdown = ""
        for image_data_index, image_data in enumerate(st.session_state.images):
            image_content = image_data["image_content"]
            image_text = image_data["text"]
            image_markdown += f"""
                <div class="gallery">
                    <a target="_blank">
                        <img src="data:image/png;base64, {image_content}" alt="{image_text}" height="350">
                    </a>
                    <div class="desc">{image_text}</div>
                </div>
            """

        if len(image_markdown) > 0:
            st.markdown(image_markdown, unsafe_allow_html=True)

    if st.session_state.results:

        st.write("### The answer to the question is:")

        for count, result in enumerate(st.session_state.results):
            if result["answer"]:
                answer = result["answer"]
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.write(
                    markdown(str(annotation(answer, "ANSWER", "#8ef"))),
                    unsafe_allow_html=True,
                )
                st.write("#### The relevant documents are:")
                for doc_i, doc in enumerate(result["document"]):
                    st.markdown(f"**Document {doc_i + 1}:** {doc}")
            else:
                st.info(
                    "🤔 &nbsp;&nbsp; fastRAG is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
                st.write("**Relevance:** ", result["relevance"])

            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)


main()
