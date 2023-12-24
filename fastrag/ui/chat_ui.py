import base64
import os

import requests
import streamlit as st
import yaml
from streamlit_chat import message

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
API_CONFIG_PATH = os.getenv("API_CONFIG_PATH", None)
SHOW_LOGO = bool(int(os.getenv("SHOW_LOGO", "1")))
PAGE_TITLE = os.getenv("PAGE_TITLE", "fastRAG Demo")
CHAT_REQUEST = "chat"
RESET_CHAT_REQUEST = "reset_chat"
UPLOAD_DOCS_REQUEST = "upload_docs"
UPLOAD_IMAGES_REQUEST = "upload_images"
DELETE_ALL_DATA = "delete_all_data"
ERROR_ANSWER = "I cannot answer that. Can you ask again?"

use_images = False
if API_CONFIG_PATH is not None:
    api_config = yaml.safe_load(open(API_CONFIG_PATH, "r"))
    use_images = "image_pipeline_file" in api_config


def upload_file():
    file_texts = []
    for uploaded_file in uploaded_files:
        bytes_data = [l.decode("utf-8") for l in uploaded_file.readlines()]
        file_texts.append(bytes_data)

    req = {"query": "upload_docs", "params": {"file_texts": file_texts}}
    response_raw = requests.post(f"{API_ENDPOINT}/{UPLOAD_DOCS_REQUEST}", json=req)
    res_json = response_raw.json()

    st.sidebar.write(f":green[{res_json['query']}!]")
    return res_json


def upload_images():
    images = []
    for uploaded_file in uploaded_images:
        uploaded_file_value = uploaded_file.getvalue()
        img_str = base64.b64encode(uploaded_file_value).decode("utf-8")

        images.append(img_str)

    req = {"query": "upload_images", "params": {"images": images}}
    response_raw = requests.post(f"{API_ENDPOINT}/{UPLOAD_IMAGES_REQUEST}", json=req)
    res_json = response_raw.json()

    st.sidebar.write(f":green[{res_json['query']}!]")
    return res_json


def delete_all_click():
    req = {"query": "delete_all", "params": {}}
    response_raw = requests.post(f"{API_ENDPOINT}/{DELETE_ALL_DATA}", json=req)
    res_json = response_raw.json()

    st.sidebar.write(f":green[{res_json['query']}!]")
    return res_json


def send_request(query, use_image_retrieval, use_retrieval):
    req = {
        "query": query,
        "params": {
            "use_retrieval": use_retrieval,
            "use_image_retrieval": use_image_retrieval,
            "inputs": st.session_state["past"],
            "outputs": [x["data"] for x in st.session_state["generated"]],
            "ret_pipe": {
                "Retriever": {"top_k": top_k_retriever},
                "Reranker": {"top_k": top_k_reranker},
            },
            "generation_kwargs": {
                "max_length": int(
                    max_new_tokens
                ),  # max_length is set to min_new_tokens in the invocation layer
                "generation_kwargs": {
                    "min_new_tokens": int(min_new_tokens),
                    "temperature": temperature,
                    "early_stopping": early_stopping,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "do_sample": do_sample,
                    "repetition_penalty": repetition_penalty,
                },
            },
            "debug": debug,
        },
    }
    response_raw = requests.post(f"{API_ENDPOINT}/{CHAT_REQUEST}", json=req)
    res_json = response_raw.json()

    return res_json


def reset_chat_request():
    req = {"query": "reset_chat"}
    response_raw = requests.post(f"{API_ENDPOINT}/{RESET_CHAT_REQUEST}", json=req)

    return response_raw


def on_input_change(use_image_retrieval, use_retrieval):
    user_input = f"{st.session_state.user_input}"
    round_count = len(st.session_state.generated)

    res_json = send_request(user_input, use_image_retrieval, use_retrieval)

    # NOTE: The modified query is located in res_json["query"]
    st.session_state.past.append(user_input)

    gen_new_state = {}
    if len(res_json["images"]) > 0:
        gen_new_state["images"] = res_json["images"]

    answer = (
        res_json["answers"][0]["answer"]
        if "answers" in res_json and len(res_json["answers"]) > 0
        else ERROR_ANSWER
    )

    gen_new_state.update({"type": "normal", "data": answer})
    st.session_state.generated.append(gen_new_state)

    if len(res_json["documents"]) > 0:
        st.session_state.docs_retrieved.append(
            {"round": round_count + 1, "query": user_input, "docs": res_json["documents"]}
        )


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    del st.session_state.docs_retrieved[:]


st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    page_icon="",
)

with open("fastrag/ui/style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("## Chat with an LLM 📚")

if SHOW_LOGO:
    st.sidebar.markdown(
        "<center><img src='https://avatars.githubusercontent.com/u/1492758?s=200&amp;v=4' width=45 height=45>&nbsp;&nbsp;&nbsp;<img src='https://github.com/IntelLabs/fastRAG/raw/056c1c643138595defbd89aa8460ea335b9cf904/assets/fastrag_header.png' width=225></center>",
        unsafe_allow_html=True,
    )


st.session_state.setdefault("past", [])
st.session_state.setdefault("generated", [])
st.session_state.setdefault("docs_retrieved", [])


st.sidebar.title("Options")

side_btn1, side_btn2 = st.sidebar.columns([1, 1])

with side_btn1:
    st.button("Delete All Data", on_click=delete_all_click)

with side_btn2:
    st.button("Clear Chat", on_click=on_btn_click)

if use_images:
    uploaded_images = st.sidebar.file_uploader("Upload a Images", accept_multiple_files=True)
    st.sidebar.button("Upload Images", on_click=upload_images)

uploaded_files = st.sidebar.file_uploader("Upload a Texutal Documents", accept_multiple_files=True)
st.sidebar.button("Upload Documents", on_click=upload_file)

####################### Retriever
top_k_retriever = st.sidebar.number_input(
    "Number of documents to initially retrieve",
    value=30,
    min_value=0,
    max_value=100,
    step=1,
)
top_k_reranker = st.sidebar.number_input(
    "Number of documents to keep after re-ranking",
    value=1,
    min_value=0,
    max_value=100,
    step=1,
)

########################

min_new_tokens = st.sidebar.number_input(
    "Min new tokens",
    value=5,
    min_value=1,
    max_value=10000,
    step=1,
)

max_new_tokens = st.sidebar.number_input(
    "Max new tokens",
    value=300,
    min_value=1,
    max_value=10000,
    step=1,
)

temperature = st.sidebar.number_input(
    "temperature",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
)

no_repeat_ngram_size = st.sidebar.number_input(
    "NGram Repeat",
    value=3,
    min_value=0,
    max_value=4,
    step=1,
)

repetition_penalty = st.sidebar.number_input(
    "repetition_penalty",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05,
)

early_stopping = st.sidebar.checkbox(
    "Early stopping",
    value=False,
)

do_sample = st.sidebar.checkbox(
    "Do Sample",
    value=False,
)


st.sidebar.write("---")

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

chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=f"{i}_user")

        cur_gen_state = st.session_state["generated"][i]
        if "images" in cur_gen_state:
            for image_j, image_content in enumerate(cur_gen_state["images"]["images"]):
                message(
                    f'<img width="100%" height="200" src="data:image/png;base64, {image_content}"/>',
                    key=f"{i}_image_{image_j}",
                    allow_html=True,
                    # is_table=True if st.session_state['generated'][i]['type']=='table' else False
                )
        message(
            cur_gen_state["data"],
            key=f"{i}",
            # allow_html=True,
            is_table=True if st.session_state["generated"][i]["type"] == "table" else False,
        )

with st.container():
    col1, col2 = st.columns([3, 1])

    with col1:
        st.text_input("", key="user_input", label_visibility="collapsed")  # on_change=
    with col2:
        btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 3])

        with btn1:
            st.button(
                "▶️",
                on_click=on_input_change,
                kwargs={"use_image_retrieval": False, "use_retrieval": False},
            )

        with btn2:
            st.button(
                "📄",
                on_click=on_input_change,
                kwargs={"use_image_retrieval": False, "use_retrieval": True},
            )

        if use_images:
            with btn3:
                st.button(
                    "🖼️",
                    on_click=on_input_change,
                    kwargs={"use_image_retrieval": True, "use_retrieval": False},
                )

            with btn4:
                st.button(
                    "📄 + 🖼️",
                    on_click=on_input_change,
                    kwargs={"use_image_retrieval": True, "use_retrieval": True},
                )


with st.expander("Documents Retrieved"):
    if len(st.session_state.docs_retrieved) > 0:
        for doc_entry in st.session_state.docs_retrieved:
            st.write(f"Documents for: {doc_entry['query']} (Round {doc_entry['round']}):")
            for doc_i, doc_current in enumerate(doc_entry["docs"]):
                st.write(f"""   Document {doc_i+1}:  {doc_current['content']}""")
    else:
        st.write("No Documents Retrieved")
