from pathlib import Path
from typing import List, Optional, Union

import torch
from haystack import ComponentError, Document, component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice, Secret


@component
class BiEncoderSimilarityRanker:
    """
    A class that represents a Bi-Encoder Similarity Ranker.

    This ranker is used to rank a list of documents based on their similarity to a given query.

    Args:
        model (Union[str, Path]): The path to the model or the model name.
        device (Optional[ComponentDevice]): The device to run the model on. Defaults to None.
        token (Optional[Secret]): The authentication token. Defaults to Secret.from_env_var("HF_API_TOKEN", strict=False).
        top_k (int): The number of top results to retrieve. Defaults to 10.
        query_prefix (str): The prefix to add to the query input. Defaults to "".
        document_prefix (str): The prefix to add to the document input. Defaults to "".
        batch_size (int): The batch size for inference. Defaults to 32.
        normalize (bool): Whether to normalize the scores. Defaults to True.
        pooling (str): The pooling strategy for encoding. Defaults to "mean".
        meta_fields_to_embed (Optional[List[str]]): The list of meta fields to embed. Defaults to None.
        embedding_separator (str): The separator for embedding meta fields. Defaults to "\n".
    """

    def __init__(
        self,
        model: Union[str, Path],
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        top_k: int = 10,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        normalize: bool = True,
        pooling: str = "mean",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        self.embedder_model = None

        self.model_name_or_path = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.top_k = top_k
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.normalize = normalize
        self.pooling = pooling
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

    def warm_up(self):
        self.embedder_model = SentenceTransformersDocumentEmbedder(
            model=self.model_name_or_path,
            device=self.device,
            token=self.token,
            prefix=self.document_prefix,  ## we'll use only for document embeddings, query is one time
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )
        self.embedder_model.warm_up()

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Runs the ranking algorithm to rank the given documents based on the query.

        Args:
            query (str): The query string.
            documents (List[Document]): The list of documents to be ranked.
            top_k (Optional[int]): The maximum number of documents to return. If not provided, the default value will be used.
        """
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if self.embedder_model is None:
            raise ComponentError(
                f"The component {self.__class__.__name__} wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        query_document = Document(content=self.query_prefix + query)
        query_vector = self.embedder_model.run([query_document])["documents"][0].embedding
        documents_with_vectors = self.embedder_model.run(documents=documents)["documents"]

        # Calculate similarity scores between query_document and documents_with_vectors
        doc_vectors = torch.tensor([doc.embedding for doc in documents_with_vectors])
        scores = torch.tensor(query_vector) @ doc_vectors.T  ## perhaps need to break it into chunks
        scores = scores.reshape(len(documents))
        # Store scores in documents_with_vectors
        for doc, score in zip(documents_with_vectors, scores.tolist()):            
            doc.score = score

        indices = scores.cpu().sort(descending=True).indices
        return {"documents": [documents_with_vectors[i.item()] for i in indices[:top_k]]}
