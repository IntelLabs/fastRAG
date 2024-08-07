from typing import Any, Dict, List, Union

import numpy as np
import torch
from tqdm import trange


class EmbedderModel:
    def __init__(
        self,
        model,
        tokenizer,
        pooling: str = "mean",
        query_prompt: str = None,
        benchmark_mode=False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.pooling = pooling
        self.max_position_embeddings = self.model.config.max_position_embeddings
        self.vocab_size = self.model.config.vocab_size
        self.query_prompt = query_prompt
        self.benchmark_mode = benchmark_mode

    def embed(self, inputs, normalize: bool = False):
        with torch.no_grad():
            if not self.benchmark_mode:
                outputs = self.model(**inputs)
            else:
                outputs = self.model(input_ids=inputs["input_ids"])
            if self.pooling == "mean":
                emb = self._mean_pooling(outputs, inputs["attention_mask"])
            elif self.pooling == "cls":
                emb = self._cls_pooling(outputs)
            else:
                raise ValueError("pooling method no supported")

            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb

    def encode_sentences(
        self,
        sentences: Union[str, List[str]],
        normalize: bool = False,
        batch_size: int = 32,
        return_as_list=False,
        convert_to_numpy=False,
        max_length=None,
    ):
        """
        returns a tensor of size len(sentences) of the embeddings
        """
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]  # type: ignore
            input_was_string = True

        if max_length is None:
            max_length = self.model.config.max_position_embeddings
        embeddings = []

        for start_index in trange(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index : start_index + batch_size]
            encoded_input = self.tokenizer(
                sentences_batch,
                padding=True,
                # padding="max_length",
                truncation=True,
                max_length=min(self.model.config.max_position_embeddings, max_length),
                return_tensors="pt",
            )
            emb = self.embed(encoded_input, normalize)
            embeddings.extend(emb)

        embeddings = torch.stack(embeddings)
        if convert_to_numpy:
            embeddings = np.asarray([emb.numpy() for emb in embeddings])
        if return_as_list:
            embeddings = [e for e in embeddings]

        if input_was_string:
            embeddings = embeddings[0]
        return embeddings

    def __call__(
        self,
        sentences: Union[str, List[str]],
        normalize: bool = False,
        batch_size: int = 32,
    ):
        return self.encode_sentences(sentences, normalize, batch_size)

    @staticmethod
    def _cls_pooling(outputs):
        if type(outputs) == dict:
            token_embeddings = outputs["last_hidden_state"]
        else:
            token_embeddings = outputs[0]
        return token_embeddings[:, 0]

    @staticmethod
    def _mean_pooling(outputs, attention_mask):
        if type(outputs) == dict:
            token_embeddings = outputs["last_hidden_state"]
        else:
            # First element of model_output contains all token embeddings
            token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class EmbedderModelMTEB(EmbedderModel):
    def encode(
        self, sentences: list[str], batch_size=32, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        return self.encode_sentences(
            sentences=sentences,
            batch_size=batch_size,
            normalize=True,
            convert_to_numpy=True,
        )

    def encode_queries(self, queries: List[str], batch_size=32, **kwargs):
        if self.query_prompt:
            sentences = [self.query_prompt + q for q in queries]
        else:
            sentences = queries
        return self.encode_sentences(
            sentences=sentences,
            batch_size=batch_size,
            normalize=True,
            convert_to_numpy=True,
        )

    def encode_corpus(
        self, corpus: Union[List[Dict[str, str]], List[str]], batch_size=32, **kwargs
    ):
        sep = " "
        if type(corpus[0]) is dict:
            sentences = [
                (
                    (doc["title"].strip() + sep + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]
        else:
            sentences = corpus
        return self.encode_sentences(
            sentences=sentences,
            batch_size=batch_size,
            normalize=True,
            convert_to_numpy=True,
        )
