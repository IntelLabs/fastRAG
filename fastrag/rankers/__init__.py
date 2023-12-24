from haystack.nodes import BaseRanker

from fastrag.utils import fastrag_timing

BaseRanker.timing = fastrag_timing

from fastrag.rankers.bi_encoder import BiEncoderRanker
from fastrag.rankers.colbert import ColBERTRanker
from fastrag.rankers.quantized_bi_encoder import QuantizedBiEncoderRanker
from fastrag.rankers.quantized_reranker import QuantizedRanker
