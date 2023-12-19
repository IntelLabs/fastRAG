from haystack.nodes import BaseRetriever

from fastrag.utils import fastrag_timing

BaseRetriever.timing = fastrag_timing

from fastrag.retrievers.colbert import ColBERTRetriever
from fastrag.retrievers.optimized import QuantizedBiEncoderRetriever
