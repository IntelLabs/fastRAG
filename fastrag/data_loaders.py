import ast
import hashlib
import json
import math
from abc import abstractmethod
from argparse import Namespace
from typing import Dict

import pandas as pd
from datasets import load_dataset
from haystack.schema import Document
from tqdm import tqdm

from fastrag.utils import AnswerGroundType, get_has_answer_data, remove_html_from_text


class BaseParser:
    def __init__(self, batch_size):
        # since document_store.write_documents *requires* batch_size, this parameter is required
        self.batch_size = batch_size

    @abstractmethod
    def __iter__(self):
        pass


def wikidpedia_data_encoder(row) -> Document:
    """encoder for wikipedia passages from HF"""
    row_id = str(hashlib.sha256(row["_id"].encode()).hexdigest())
    return Document(
        content=str(row["passage_text"]),
        id=row_id,
        meta={"title": str(row["article_title"])},
    )


def wiki_odqa_tasks_encoder(doc) -> Dict:
    """encoder for NQ/TQA ODQA dataset"""
    query = str(doc["query"])
    answers = list(set(doc["answers"]))
    return {"question": query, "answers": answers}


def squad_odqa_encoder(doc) -> Dict:
    """encoder for squad ODQA dataset"""
    query = str(doc["query"])
    answers = list(set(ast.literal_eval("".join(doc["answers"]))))
    return {"question": query, "answers": answers}


def wikipedia_hf_encoder(doc) -> Document:
    """encoder for wikipedia dataset from HF datasets"""
    return Document(content=doc["text"], id=doc["docid"], meta={"title": doc["title"]})


def hf_id_title_text(doc) -> Document:
    """encoder for BeIR/NQ dataset from HF datasets"""
    return Document(content=doc["text"], id=doc["_id"], meta={"title": doc["title"]})


def hf_id_title_text_concat(doc) -> Document:
    """encoder for BeIR/NQ dataset from HF datasets"""
    return Document(content=(doc["title"] + "\n" + doc["text"]), id=doc["_id"])


encoding_methods = {
    "wikidpedia_data": wikidpedia_data_encoder,
    "squad_odqa": squad_odqa_encoder,
    "nq_decoder": wiki_odqa_tasks_encoder,
    "tqa_decoder": wiki_odqa_tasks_encoder,
    "wikipedia_hf": wikipedia_hf_encoder,
    "hf_id_title_text": hf_id_title_text,
    "hf_id_title_text_concat": hf_id_title_text_concat,
}


class HFDatasetLoader(BaseParser):
    def __init__(
        self,
        dataset_info,
        encoding_method,
        batch_size=1,
    ):
        super().__init__(batch_size)

        # load the dataset from HF
        self.data = load_dataset(**dataset_info)
        self.length = len(self.data)
        self.chunks = (self.length // self.batch_size) + 1
        self.encode_fn = encoding_methods[encoding_method]

    def __iter__(self):
        for i in tqdm(
            list(range(self.chunks)),
            desc="Data chunks",
        ):
            end_size = min((i + 1) * self.batch_size, self.length)
            docs = [self.encode_fn(self.data[j]) for j in range(i * self.batch_size, end_size)]
            yield docs


def encode_stackoverflow(row, fields_to_construct_from, id_field):
    """
    Encodes the content in the stackoverflow datasets, to create documents for the store.
    fields_to_construct_from: The field to use for creating the text.
    id_field: The field that corresponds with the id of the content.
    """
    contents = []
    for field in fields_to_construct_from:
        content = str(row[field])
        if field != "title":
            content = remove_html_from_text(content)
        contents.append(content)
    overall_text = ". ".join(contents)
    return Document(content=overall_text, id=str(id_field))


def encode_wikipedia_text(row):
    return Document(content=str(row["text"]), id=str(row["id"]))


def encode_wikipedia_text_and_title(row):
    return Document(
        content=str(row["text"]),
        id=str(row["id"]),
        meta={"title": str(row["title"])},
    )


def encode_wikipedia_title_only(row):
    return Document(content=str(row["title"]), id=str(row["id"]))


def encode_stackoverflow_answer(row):
    return encode_stackoverflow(
        row, fields_to_construct_from=["answer_body"], id_field="AcceptedAnswerId"
    )


def encode_stackoverflow_body_answer(row):
    return encode_stackoverflow(
        row, fields_to_construct_from=["body", "answer_body"], id_field="id"
    )


def encode_stackoverflow_body(row):
    return encode_stackoverflow(row, fields_to_construct_from=["body"], id_field="id")


def encode_question_answer_test_csv(row):
    return {"question": str(row[0]), "answers": list(set(ast.literal_eval(row[1])))}


def encode_evaluation_question_content(row):
    return {
        "question": str(row["title"]),
        "answers": [str(row["id"])],
        "answer_type": AnswerGroundType.ID,
    }


row_parser_functions = {
    "text": encode_wikipedia_text,
    "text_and_title": encode_wikipedia_text_and_title,
    "only_title": encode_wikipedia_title_only,
    "question_answer_test_csv": encode_question_answer_test_csv,
    "stackoverflow_answer": encode_stackoverflow_answer,
    "stackoverflow_body_answer": encode_stackoverflow_body_answer,
    "stackoverflow_body": encode_stackoverflow_body,
    "evaluation_question_content": encode_evaluation_question_content,
}


class CSVFileLoader(BaseParser):
    def __init__(
        self,
        filepath,
        encoding_method,
        delimiter="\t",
        batch_size=1,
    ):
        super().__init__(batch_size)

        self.filepath = filepath
        self.file_length = _get_file_length(self.filepath)
        self.iterations_count = math.ceil(self.file_length / self.batch_size)
        self.delimiter = delimiter
        self.encode_fn = row_parser_functions[encoding_method]

    def __iter__(self):
        for df in tqdm(
            pd.read_csv(
                filepath_or_buffer=self.filepath,
                chunksize=self.batch_size,
                delimiter=self.delimiter,
            ),
            total=self.iterations_count,
            desc="Document chunks",
        ):
            docs = [self.encode_fn(row) for _, row in df.iterrows()]
            yield docs


def _get_file_length(filepath):
    with open(filepath) as f:
        line_count = 0
        for line in f:
            line_count += 1
    return line_count
