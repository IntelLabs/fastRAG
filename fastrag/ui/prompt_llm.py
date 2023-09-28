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

import streamlit as st
from annotated_text import annotated_text
from markdown import markdown

from fastrag.ui.utils import API_ENDPOINT, display_runtime_plot, haystack_is_ready, query

DEFAULT_PROMPT = """Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words. \n\n Context: {join(documents)} \n\n Question: {query} \n\n Answer:"""
pipeline_path = os.getenv("PIPELINE_PATH", None)

if(pipeline_path is not None):
    import yaml

    with open(pipeline_path, "r") as stream:
        pipeline_file = yaml.safe_load(stream)
    prompt_template_components = [x for x in pipeline_file['components'] if x['type'] == "PromptTemplate"]
    DEFAULT_PROMPT = prompt_template_components[0]['params']['prompt']


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def clean_markdown(string):
    return string.replace("$", "\$").replace(":", "\:")


def main():
    st.set_page_config(
        page_title="fastRAG Demo",
        layout="wide",
        page_icon="",
    )

    with open("fastrag/ui/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Persistent state
    set_state_if_absent("question", None)
    set_state_if_absent("answer", None)
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

    # Title
    st.markdown("## Retrieve and Generate with a LLM 📚")
    st.sidebar.markdown(
        "<center><img src='https://avatars.githubusercontent.com/u/1492758?s=200&amp;v=4' width=45 height=45>&nbsp;&nbsp;&nbsp;<img src='https://github.com/IntelLabs/fastRAG/raw/056c1c643138595defbd89aa8460ea335b9cf904/assets/fastrag_header.png' width=225></center>",
        unsafe_allow_html=True,
    )
    # Sidebar
    st.sidebar.title("Options")
    top_k_retriever = st.sidebar.number_input(
        "Documents to retrieve from the index",
        value=50,
        min_value=1,
        max_value=100,
        step=1,
        # on_change=reset_results,
    )
    top_k_reranker = st.sidebar.number_input(
        "Documents document to re-rank",
        value=3,
        min_value=1,
        max_value=50,
        step=1,
        # on_change=reset_results,
    )
    min_new_tokens = st.sidebar.number_input(
        "Min new tokens",
        value=20,
        min_value=1,
        max_value=100,
        step=1,
        # on_change=reset_results
    )

    max_new_tokens = st.sidebar.number_input(
        "Max new tokens",
        value=1000,
        min_value=1,
        max_value=10000,
        step=1,
        # on_change=reset_results
    )

    decode_mode = st.sidebar.selectbox(
        "Decoding mode",
        options=["Beam", "Greedy"],
        index=1,
        # on_change=reset_results
    )

    temperature = st.sidebar.slider(
        "temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        # on_change=reset_results
    )

    top_p = st.sidebar.slider(
        "top_p",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        # on_change=reset_results
    )

    beams = st.sidebar.number_input(
        "Number of beams",
        value=4,
        min_value=1,
        max_value=4,
        step=1,
        # on_change=reset_results
    )

    early_stopping = st.sidebar.checkbox(
        "Early stopping",
        value=True,
        # on_change=reset_results
    )

    st.sidebar.write("---")

    show_runtime = st.sidebar.checkbox("Show components runtime")
    debug = st.sidebar.checkbox("Show debug info")

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
    examples = ""

    with st.expander("Customize Prompt"):
        prompt_template = st.text_area(
            label="Prompt template", max_chars=500, value=DEFAULT_PROMPT, height=150
        )

    question = st.text_input(label="Query", max_chars=1000, value=examples)

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

        pipeline_params_dict = {
            "input_prompt": {"prompt": prompt_template},
            "generation_kwargs": {
                "min_new_tokens": int(min_new_tokens),
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "top_p": top_p,
                "num_beams": beams,
                "early_stopping": early_stopping,
            },
        }
        if "Greedy" in decode_mode:
            pipeline_params_dict["generation_kwargs"]["num_beams"] = 1
            pipeline_params_dict["generation_kwargs"]["do_sample"] = False
        elif "Beam" in decode_mode:
            pipeline_params_dict["generation_kwargs"]["do_sample"] = True
        pipeline_params_dict["generation_kwargs"]["min_length"] = None
        pipeline_params_dict["generation_kwargs"]["max_length"] = None

        with st.spinner("Searching through documents and generating answers ... \n "):
            try:
                st.session_state.clear()
                (
                    st.session_state.results,
                    st.session_state.raw_json,
                    st.session_state.images,
                    st.session_state.relations,
                ) = query(
                    question,
                    top_k_retriever=top_k_retriever,
                    top_k_reranker=top_k_reranker,
                    pipeline_params_dict=pipeline_params_dict,
                    debug=True,
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

    if st.session_state.raw_json is not None:
        retrieved_docs = st.session_state.raw_json["_debug"]["Reranker"]["output"]["documents"]

    if st.session_state.images or st.session_state.results:
        st.write("### Response")

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
        for count, result in enumerate(st.session_state.results):
            if result["answer"]:
                answer = result["answer"]
                annotated_text((answer, "Answer"))
                st.write("___")
                st.write("#### Supporting documents")
                for doc_i, doc in enumerate(retrieved_docs):
                    doc_prefix = doc['meta'].get('title', f"Document {doc_i+1}")
                    st.write(
                        f"**{doc_prefix}:** {clean_markdown(doc.get('content'))}"
                    )
            else:
                st.info(
                    "🤔 &nbsp;&nbsp; fastRAG is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
        if debug:
            st.write("___")
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)


main()
