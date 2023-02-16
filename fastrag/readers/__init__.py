from typing import List, Optional, Union

from haystack.nodes import BaseGenerator, BaseReader
from haystack.schema import Document, MultiLabel

from fastrag.readers.FiD import FiDReader
from fastrag.readers.T5 import T5Reader
from fastrag.utils import fastrag_timing

BaseReader.timing = fastrag_timing

# Patch BaseGenerator timing functionality
BaseGenerator.query_count = 0
BaseGenerator.query_time = 0
BaseGenerator.timing = fastrag_timing


def run(self, query: str, documents: List[Document], top_k: Optional[int] = None, labels: Optional[MultiLabel] = None, add_isolated_node_eval: bool = False):  # type: ignore
    self.query_count += 1
    predict = self.timing(self.predict, "query_time")
    if documents:
        results = predict(query=query, documents=documents, top_k=top_k)
    else:
        results = {"answers": []}

    # run evaluation with "perfect" labels as node inputs to calculate "upper bound" metrics for just this node
    if add_isolated_node_eval and labels is not None:
        relevant_documents = list(
            {label.document.id: label.document for label in labels.labels}.values()
        )
        results_label_input = self.predict(query=query, documents=relevant_documents, top_k=top_k)
        results["answers_isolated"] = results_label_input["answers"]

    return results, "output_1"


def run_batch(  # type: ignore
    self,
    queries: List[str],
    documents: Union[List[Document], List[List[Document]]],
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    self.query_count += len(queries)
    predict_batch = self.timing(self.predict_batch, "query_time")
    results = self.predict_batch(
        queries=queries, documents=documents, top_k=top_k, batch_size=batch_size
    )
    return results, "output_1"


BaseGenerator.run = run
BaseGenerator.run_batch = run_batch
