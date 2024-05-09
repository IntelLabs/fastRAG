import ast
import hashlib
import math
from abc import abstractmethod
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from haystack import Document
from tqdm import tqdm

from fastrag.utils import AnswerGroundType, remove_html_from_text

try:
    import nltk

    nltk.download("punkt")
    nltk_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
except ImportError as e:
    print("nltk is not installed")


class BaseParser:
    def __init__(self, batch_size, limit=None):
        # since document_store.write_documents *requires* batch_size, this parameter is required
        self.batch_size = batch_size
        self.limit = limit

    @abstractmethod
    def __iter__(self):
        pass


def text_encoder(row) -> Document:
    """encoder for text passages"""
    return Document(content=str(row["text"]))


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


def pubmedQA_hf_encoder(doc) -> Document:
    """encoder for pubmedQA dataset from HF datasets

    Splits the original passage into subpassages 5 sentences each.
    """
    sentences = nltk_tokenizer.tokenize(doc["context"])
    joined_passages = sentences_to_passages(sentences, sentences_per_passage=5)

    return [
        Document(content=s, id=f"{str(doc['document_id'])}_{s_idx}")
        for s_idx, s in enumerate(joined_passages)
    ]


def hf_id_title_text(doc) -> Document:
    """encoder for BeIR/NQ dataset from HF datasets"""
    return Document(content=doc["text"], id=doc["_id"], meta={"title": doc["title"]})


def hf_id_title_text_concat(doc) -> Document:
    """encoder for BeIR/NQ dataset from HF datasets"""
    return Document(content=(doc["title"] + "\n" + doc["text"]), id=doc["_id"])


def sentences_to_passages(sens, sentences_per_passage=3):
    sens_count = len(sens)

    sens_batch_count = (sens_count // sentences_per_passage) + 1

    passages = []
    for sen_batch_index in range(sens_batch_count):
        sens_current_batch = sens[
            sen_batch_index * sentences_per_passage : (sen_batch_index + 1) * sentences_per_passage
        ]
        if len(sens_current_batch) == 0:
            continue

        passage_current = " ".join(sens_current_batch)
        passages.append(passage_current)

    return passages


def wikipedia_hf_multisentence_encoder(doc) -> List[Document]:
    """encoder for wikipedia dataset from HF datasets"""

    sentences = nltk_tokenizer.tokenize(str(doc["text"]))

    joined_passages = sentences_to_passages(sentences)

    return [
        Document(content=s, id=f"{str(doc['id'])}_{s_idx}", meta={"title": str(doc["title"])})
        for s_idx, s in enumerate(joined_passages)
    ]


encoding_methods = {
    "text": text_encoder,
    "wikidpedia_data": wikidpedia_data_encoder,
    "squad_odqa": squad_odqa_encoder,
    "nq_decoder": wiki_odqa_tasks_encoder,
    "tqa_decoder": wiki_odqa_tasks_encoder,
    "wikipedia_hf": wikipedia_hf_encoder,
    "wikipedia_hf_multisentence": wikipedia_hf_multisentence_encoder,
    "pubmedqa_hf": pubmedQA_hf_encoder,
    "hf_id_title_text": hf_id_title_text,
    "hf_id_title_text_concat": hf_id_title_text_concat,
}


class HFDatasetLoader(BaseParser):
    def __init__(
        self,
        dataset_info,
        encoding_method,
        batch_size=1,
        limit=None,
    ):
        super().__init__(batch_size, limit)

        # load the dataset from HF
        self.data = load_dataset(**dataset_info)
        self.length = self.limit or len(self.data)
        self.chunks = (self.length // self.batch_size) + 1
        self.encode_fn = encoding_methods[encoding_method]

    def __iter__(self):
        for i in tqdm(
            list(range(self.chunks)),
            desc="Data chunks",
        ):
            docs = self.process(i)
            yield docs

    def process(self, i):
        """Given a batch number i, returns the processed batch.
        Useful for restarting a job from the ith batch."""

        end_size = min((i + 1) * self.batch_size, self.length)
        docs = []
        for j in range(i * self.batch_size, end_size):
            encoding_results = self.encode_fn(self.data[j])
            if isinstance(encoding_results, list):
                docs.extend(encoding_results)
            else:
                docs.append(encoding_results)
        return docs


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


def encode_wikipedia_json(row):
    return Document(content=row["text"], id=str(row["id"]), meta={"title": row["title"]})


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
    "wikipedia_local_json": encode_wikipedia_json,
}


class CSVFileLoader(BaseParser):
    def __init__(
        self,
        filepath,
        encoding_method,
        delimiter="\t",
        batch_size=1,
        limit=None,
    ):
        super().__init__(batch_size, limit)

        self.filepath = filepath
        self.file_length = self.limit or _get_file_length(self.filepath)
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


class JSONFileLoader(BaseParser):
    def __init__(
        self,
        filepath,
        encoding_method,
        batch_size=1,
    ):
        super().__init__(batch_size)

        self.filepath = filepath
        self.file_length = _get_file_length(self.filepath)
        self.iterations_count = math.ceil(self.file_length / self.batch_size)
        self.encode_fn = row_parser_functions[encoding_method]

    def __iter__(self):
        for df in tqdm(
            pd.read_json(
                self.filepath,
                chunksize=self.batch_size,
                lines=self.filepath.endswith("jsonl"),
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
