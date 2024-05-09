import inspect
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from haystack import Document, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install .[colbert]' to install the ColBERT library.") as colbert_import:
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig


logger = logging.getLogger(__name__)


class PLAIDDocumentStore:
    """
    Store for ColBERT v2 with PLAID indexing.

    Parameters:

    index_path: directory containing PLAID index files.
    checkpoint_path: directory containing ColBERT checkpoint model files.
    collection_path: a csv/tsv data file of the form (id,content), no header line.

    create: whether to create a new index or load an index from disk. Default: False.

    nbits: number of bits to quantize the residual vectors. Default: 2.
    kmeans_niters: number of kmeans clustering iterations. Default: 1.
    gpus: number of GPUs to use for indexing. Default: 0.
    rank: number of ranks to use for indexing. Default: 1.
    doc_maxlen: max document length. Default: 120.
    query_maxlen: max query length. Default: 60.

    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        params = inspect.signature(self.__init__).parameters
        init_params = {k: getattr(self, k) for k in params}
        return default_to_dict(
            self,
            **init_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PLAIDDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns the number of documents stored.
        """
        return len(self.docs)

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.
        """
        pass

    def write_documents(self, documents: List[Document], **kwargs) -> int:
        """
        Writes (or overwrites) documents into the DocumentStore, return the number of documents that was written.
        """
        raise Exception(
            "PLAIDDocumentStore can only be used as a read-only store. A new index is needed for adding/changing documents"
        )

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the DocumentStore.
        """
        pass

    def __init__(
        self,
        index_path,
        checkpoint_path,
        collection_path,
        create=False,
        nbits=2,
        gpus=0,
        ranks=1,
        doc_maxlen=120,
        query_maxlen=60,
        kmeans_niters=4,
    ):
        colbert_import.check()
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path
        self.collection_path = collection_path
        self.nbits = nbits
        self.gpus = gpus
        self.ranks = ranks
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.kmeans_niters = kmeans_niters

        if create:
            self._create_index()

        self.docs = pd.read_csv(
            collection_path, sep="\t" if collection_path.endswith(".tsv") else ",", header=None
        )
        self.titles = len(self.docs.columns) > 2
        self._load_index()

    def _load_index(self):
        """Load PLAID index from the paths given to the class and initialize a Searcher object."""
        with Run().context(
            RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            self.store = Searcher(
                index="", collection=self.collection_path, checkpoint=self.checkpoint_path
            )

        logger.info("Loaded PLAIDDocumentStore index")

    def _create_index(self):
        """Generate a PLAID index from a given ColBERT checkpoint.

        Given a checkpoint and a collection of documents, an Indexer object will be created.
        The index will then be generated, written to disk at `index_path` and finally it
        will be loaded.
        """

        with Run().context(
            RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                kmeans_niters=self.kmeans_niters,
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index("", collection=self.collection_path, overwrite=True)

        logger.info("Created PLAIDDocumentStore Index.")

    def query(self, query_str, top_k=10) -> List[Document]:
        """
        Query the Colbert v2 + Plaid store.

        Returns: list of Haystack documents.
        """

        doc_ids, _, scores = self.store.search(text=query_str, k=top_k)

        documents = [
            Document.from_dict(
                {
                    "content": self.docs.iloc[_id][1],
                    "id": _id,
                    "score": score,
                    "meta": {"title": self.docs.iloc[_id][2] if self.titles else None},
                }
            )
            for _id, score in zip(doc_ids, scores)
        ]

        return documents
