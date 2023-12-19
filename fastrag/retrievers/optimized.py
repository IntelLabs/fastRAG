import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from haystack.document_stores.base import BaseDocumentStore
from haystack.modeling.utils import initialize_device_settings  # pylint: disable=ungrouped-imports
from haystack.nodes.retriever.dense import DenseRetriever
from haystack.schema import Document, FilterType
from tqdm import tqdm

from fastrag.rankers import QuantizedBiEncoderRanker

logger = logging.getLogger(__name__)


class QuantizedBiEncoderRetriever(DenseRetriever):
    """
    An optimized retriever that uses a bi-encoder embedder for embeddings queries and documents.
    Uses CPU backends for optimized performance.
    """

    def __init__(
        self,
        embedding_model: str,
        document_store: Optional[BaseDocumentStore] = None,
        batch_size: int = 32,
        max_seq_len: int = 512,
        pooling_strategy: str = "mean",  # "mean" or "cls"
        top_k: int = 10,
        progress_bar: bool = True,
        scale_score: bool = True,
        embed_meta_fields: Optional[List[str]] = None,
        query_prompt: Optional[str] = None,
        document_prompt: Optional[str] = None,
        pad_to_max: Optional[bool] = False,
    ):
        super().__init__()

        self.devices, _ = initialize_device_settings(devices="cpu", use_cuda=False, multi_gpu=False)

        self.embedding_model = embedding_model
        self.document_store = document_store
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.scale_score = scale_score
        # self.embed_meta_fields = embed_meta_fields
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.pad_to_max = pad_to_max

        self.embedder = QuantizedBiEncoderRanker(
            model_name_or_path=embedding_model,
            top_k=top_k,
            batch_size=batch_size,
            scale_score=scale_score,
            pooling=pooling_strategy,
            progress_bar=progress_bar,
            query_prompt=query_prompt,
            document_prompt=document_prompt,
            pad_to_max=pad_to_max,
        )
        logger.info("Init retriever using embeddings of model %s", embedding_model)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        # for backward compatibility: cast pure str input
        if isinstance(queries, str):
            queries = [queries]
        assert isinstance(
            queries, list
        ), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        query_batches = self._get_batches(queries, self.batch_size)
        pb = tqdm(total=len(queries), disable=not self.progress_bar, desc="Encoding queries")
        query_vectors = []
        for batch in query_batches:
            _q = [q if not self.query_prompt else self.query_prompt + q for q in batch]
            query_tok = self.embedder.transformer_tokenizer(
                _q, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len
            ).to(self.devices[0])
            q_vecs = self.embedder(query_tok)
            query_vectors.extend(q_vecs)
            pb.update(len(batch))
        pb.close()
        query_vectors = torch.stack(query_vectors).reshape(len(queries), -1)
        return query_vectors.detach().cpu().numpy()

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings of documents, one per input document, shape: (documents, embedding_dim)
        """
        document_batches = self._get_batches(documents, self.batch_size)
        pb = tqdm(total=len(documents), disable=not self.progress_bar, desc="Encoding documents")
        document_vectors = []
        for batch in document_batches:
            _d = [
                d.content if not self.document_prompt else self.document_prompt + d.content
                for d in batch
            ]
            documents_tok = self.embedder.transformer_tokenizer(
                _d,
                padding="max_length" if self.pad_to_max else True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            ).to(self.devices[0])
            d_vecs = self.embedder(documents_tok)
            document_vectors.extend(d_vecs)
            pb.update(len(batch))
        pb.close()
        document_vectors = torch.stack(document_vectors).reshape(len(documents), -1)
        return document_vectors.detach().cpu().numpy()

    def run_indexing(self, documents: List[Document]):
        embeddings = self.embed_documents(documents)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        output = {"documents": documents}
        return output, "output_1"

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the __init__ is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])
        documents = document_store.query_by_embedding(
            query_emb=query_emb[0],
            top_k=top_k,
            filters=filters,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )

        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score

        query_embs: List[np.ndarray] = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(queries=batch))
        print(query_embs)
        documents = document_store.query_by_embedding_batch(
            query_embs=query_embs,
            top_k=top_k,
            filters=filters,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )

        return documents
