from pathlib import Path
from typing import List, Optional, Union

import torch
from haystack import Document, component
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install .[colbert]' to install the ColBERT library.") as colbert_import:
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.search.strided_tensor_core import StridedTensorCore


@component
class ColBERTRanker:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        top_k: int = 10,
        query_maxlen: int = 120,
        doc_maxlen: int = 120,
        dim: int = 128,
        use_gpu: bool = False,
    ):
        colbert_import.check()
        self.top_k = top_k

        self.config = ColBERTConfig(
            checkpoint=checkpoint_path,
            ncells=2,
            nbits=2,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            dim=dim,
        )

        self.use_gpu = use_gpu
        self.to_cpu = not use_gpu

        self.checkpoint = Checkpoint(checkpoint_path, self.config)
        if self.use_gpu:
            self.checkpoint.model.cuda()

        self.checkpoint.query_tokenizer.query_maxlen = query_maxlen
        self.checkpoint.query_tokenizer.doc_maxlen = doc_maxlen

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if top_k is None:
            top_k = self.top_k

        query_emb = self.checkpoint.queryFromText([query], bsize=64, to_cpu=self.to_cpu)
        if self.use_gpu:
            query_emb = query_emb.half()

        tensor, lengths = self.checkpoint.docFromText(
            [d.content for d in documents], keep_dims="flatten", bsize=64, to_cpu=self.to_cpu
        )
        tensor, masks = StridedTensorCore(tensor, lengths, use_gpu=self.use_gpu).as_padded_tensor()
        if not self.use_gpu:
            tensor, masks = tensor.float(), masks.float()

        # MaxSim implementation. Returns [#queries, #documents] tensor.
        # Output identical to colbert.modeling.colbert.colbert_score
        # however scores are different than when calculated at query time.
        # It's due to quantization differences when recovering the embeddings.
        scores = torch.einsum("bwd,qvd -> bqwv", query_emb, tensor * masks).max(-1).values.sum(-1)
        for doc, score in zip(documents, scores.tolist()):
            doc.score = score

        indices = scores.cpu().sort(descending=True).indices[0]
        return {"documents": [documents[i.item()] for i in indices[:top_k]]}
