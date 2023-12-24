from typing import Any, Dict, List, Optional, Tuple

from haystack import BaseComponent
from haystack.schema import Document


class DocumentLister(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
    ):
        super().__init__()

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        context_text = ""
        for doc_index, doc in enumerate(documents):
            context_text += f"Paragraph {doc_index+1}: {doc.content}\n\n"

        joined_doc = Document(content=context_text)

        results = {
            "documents": [joined_doc],
            "invocation_context": {
                "query": query,
                "documents": [joined_doc],
            },
        }

        return results, "output_1"

    def run_batch(  # type: ignore
        self,
        query: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        return self.run(
            query=query,
            documents=documents,
            invocation_context=invocation_context,
        )
