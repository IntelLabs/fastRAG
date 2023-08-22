from inspect import currentframe, getframeinfo
from typing import List, Optional, Union

import torch
from haystack.nodes import Seq2SeqGenerator
from haystack.schema import Document

import fastrag

logger = fastrag.utils.init_logger(__name__)


class T5Reader(Seq2SeqGenerator):
    def __init__(
        self,
        model_name_or_path: str,
        top_k: int = 1,
        max_length: int = 200,
        min_length: int = 2,
        num_beams: int = 8,
        use_gpu: bool = True,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        input_converter_tokenizer_max_len: int = 256,
        input_converter_mode: str = "qa",
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            input_converter=_T5Converter(input_converter_tokenizer_max_len, input_converter_mode),
            top_k=top_k,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            use_gpu=use_gpu,
            progress_bar=progress_bar,
            use_auth_token=use_auth_token,
            devices=devices,
        )
        logger.info(f"Init T5 reader. Mode: {input_converter_mode}")

    # a function to execute predict directly
    def predict(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 1,
        max_length: int = None,
        min_length: int = None,
    ) -> dict:
        logger.info(
            f"in method: {getframeinfo(currentframe()).function}, line: {getframeinfo(currentframe()).lineno}"
        )
        if max_length:
            logger.info(f"Changing max_length to: {max_length}")
            self.max_length = max_length
        if min_length:
            logger.info(f"Changing min_length to: {min_length}")
            self.min_length = min_length
        logger.info(f"self.min_length: {self.min_length}")
        logger.info(f"self.max_length: {self.max_length}")
        if isinstance(documents, list) and all([isinstance(doc, Document) for doc in documents]):
            return super().predict(
                query=query, documents=documents, top_k=top_k
            )  # we use run() and not predict(), as run returns
        else:  # answer according to haystack pipeline node requirements
            raise Exception(f"input documents is not supported!!! {documents}")

    # a function to execute predict as part of haysack pipeline
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 1,
        max_length: int = None,
        min_length: int = None,
    ) -> dict:
        logger.info(
            f"in method: {getframeinfo(currentframe()).function}, line: {getframeinfo(currentframe()).lineno}"
        )
        if max_length:
            logger.info(f"Changing max_length to: {max_length}")
            self.max_length = max_length
        if min_length:
            logger.info(f"Changing min_length to: {min_length}")
            self.min_length = min_length
        logger.info(f"self.min_length: {self.min_length}")
        logger.info(f"self.max_length: {self.max_length}")
        if isinstance(documents, list) and all([isinstance(doc, Document) for doc in documents]):
            return super().run(
                query=query, documents=documents, top_k=top_k
            )  # we use run() and not predict(), as run returns
        else:  # answer according to haystack pipeline node requirements
            raise Exception(f"input documents is not supported!!! {documents}")

    # a function to execute predict_batch directly
    def predict_batch(
        self,
        queries: List[str],
        documents: List[List[Document]],
        top_k: int = None,
        batch_size: int = None,
        max_length: int = None,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            input_converter=_T5Converter(input_converter_tokenizer_max_len, input_converter_mode),
            top_k=top_k,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            use_gpu=use_gpu,
            progress_bar=progress_bar,
            use_auth_token=use_auth_token,
            devices=devices,
        )
        if max_length:
            logger.info(f"Changing max_length to: {max_length}")
            self.max_length = max_length
        logger.info(f"self.max_length: {self.max_length}")
        return super().predict_batch(queries=queries, documents=documents, top_k=top_k)

    # a function to execute predict_batch as part of haysack pipeline
    def run_batch(
        self,
        queries: List[str],
        documents: List[List[Document]],
        top_k: int = None,
        batch_size: int = None,
        max_length: int = None,
        min_length: int = None,
    ):
        assert len(queries) == len(documents)
        logger.info(
            f"in method: {getframeinfo(currentframe()).function}, line: {getframeinfo(currentframe()).lineno}"
        )
        if max_length:
            logger.info(f"Changing max_length to: {max_length}")
            self.max_length = max_length
        if min_length:
            logger.info(f"Changing min_length to: {min_length}")
            self.min_length = min_length
        logger.info(f"self.min_length: {self.min_length}")
        logger.info(f"self.max_length: {self.max_length}")
        return super().run_batch(queries=queries, documents=documents, top_k=top_k)


class _T5Converter:
    def __init__(self, tokenizer_max_len, mode):
        self.mode = mode
        self.max_len = tokenizer_max_len  # the maximal length of input's tokens series
        logger.info(
            f"tokenizer max length is:{self.max_len}. in method: {getframeinfo(currentframe()).function}, line: {getframeinfo(currentframe()).lineno}"
        )

    def __call__(self, tokenizer, query: str, documents: List[Document], top_k=None):
        logger.info("In _T5Converter.__call__")
        if tokenizer.model_max_length > self.max_len:
            logger.info(
                f"Warning!!! You set {self.max_len} as max len for the tokenizer, which is smaller than model_max_length {tokenizer.model_max_length}. in method: {getframeinfo(currentframe()).function}, line: {getframeinfo(currentframe()).lineno}"
            )

        preprocessed_doc = ". ".join([doc.content for doc in documents])
        # preprocessed_doc = [doc.content for doc in documents]

        if self.mode == "qa":
            prompt_with_content = f"Answer the question given the context. Question: {query} Context: {preprocessed_doc}"
        elif self.mode == "translation":
            prompt_with_content = (
                f"{query}: {preprocessed_doc}"  # for translation use this as tokenizer input
            )
        elif self.mode == "summarization":
            prompt_with_content = (
                f"summarize: {preprocessed_doc}"  # for summarization use this as tokenizer input
            )
        else:
            raise Exception(f"ERROR! mode: {self.mode} is not supported.")

        return tokenizer(
            [prompt_with_content],
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
        )
