from haystack.nodes import BaseRetriever

from fastrag.utils import fastrag_timing, safe_import

BaseRetriever.timing = fastrag_timing

ColBERTRetriever = safe_import("fastrag.retrievers.colbert", "ColBERTRetriever")
