import logging
from typing import List, Optional

from haystack import Document, component

from fastrag.stores import PLAIDDocumentStore

logger = logging.getLogger(__name__)


@component
class ColBERTRetriever:
    def __init__(
        self,
        document_store: PLAIDDocumentStore,
        top_k: int = 10,
    ):
        self.document_store = document_store
        self.top_k = top_k

        logger.info(f"Initialize retriever using the store: {document_store}")

    @component.output_types(documents=List[Document])
    def run(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Document]:
        top_k = top_k or self.top_k
        documents = self.document_store.query(query_str=query, top_k=top_k)
        return {"documents": documents}
