from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Union

import torch
from haystack.lazy_imports import LazyImport
from haystack.nodes.ranker import BaseRanker
from haystack.schema import Document
from transformers import AutoTokenizer

with LazyImport(
    "No installation of intel_extension_for_transformers found. Please install it `pip install intel-extension-for-transformers"
) as ipex_backend_import:
    from intel_extension_for_transformers.backends.neural_engine.compile import compile


class QuantizedRanker(BaseRanker):
    def __init__(
        self,
        onnx_file_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        top_k: int = 10,
        max_length: int = 256,
    ):
        ipex_backend_import.check()
        self.top_k = top_k
        self.model = compile(onnx_file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length

    @staticmethod
    def create_batch(batch, max_length, tokenizer):
        texts = [[] for _ in range(len(batch[0].texts))]

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

        tokenized = tokenizer(
            *texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=max_length
        )

        return [
            tokenized["input_ids"].numpy().astype("int32"),
            tokenized["token_type_ids"].numpy().astype("int32"),
            tokenized["attention_mask"].numpy().astype("int32"),
        ]

    def get_prediction_scores(self, batch):
        predictions = self.model.inference(batch)

        pred_key = list(predictions.keys())[0]
        preds = predictions[pred_key]

        cross_scores = preds.reshape(-1).tolist()
        return cross_scores

    def predict_on_pairs(self, query, corpus_sentences):
        cross_inp = [Namespace(texts=[query, sent]) for sent in corpus_sentences]
        batch = QuantizedRanker.create_batch(cross_inp, self.max_length, self.tokenizer)

        pred = self.get_prediction_scores(batch)

        return torch.tensor(pred)

    def predict(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if top_k is None:
            top_k = self.top_k

        corpus_sentences = [d.content for d in documents]

        scores = self.predict_on_pairs(query, corpus_sentences)
        indices = scores.cpu().sort(descending=True).indices
        return [documents[i] for i in indices[:top_k]]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[Document]]:
        if top_k is None:
            top_k = self.top_k

        predictions = []

        if isinstance(documents[0], list):
            assert len(queries) == len(
                documents
            ), "Lists of documents should match number of queries."
            for query, q_documents in zip(queries, documents):
                predictions.append(self.predict(query, q_documents, top_k))

        else:
            for q in queries:
                prediction = self.predict(q, documents, top_k=top_k)
                predictions.append(prediction)

        return predictions
