from haystack.nodes import BaseReader

from fastrag.utils import fastrag_timing

BaseReader.timing = fastrag_timing
