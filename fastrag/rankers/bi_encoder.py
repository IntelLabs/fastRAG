from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

from haystack.errors import HaystackError
from haystack.lazy_imports import LazyImport
from haystack.nodes.ranker import BaseRanker
from haystack.schema import Document
from tqdm import tqdm

with LazyImport("Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from haystack.modeling.utils import (  # pylint: disable=ungrouped-imports
        initialize_device_settings,
    )
    from torch.nn import DataParallel
    from transformers import AutoModel, AutoTokenizer


class BiEncoderRanker(BaseRanker):
    """
    Bi-encoder re-ranker model based on Encoder models (like BERT) that can be used for both queries and documents.
    Has higher efficiency when compared to cross-encoders with a slight reduction in performance.

    Re-Ranking can be used on top of a retriever to boost the performance for document search.
    This is particularly useful if the retriever has a high recall but is bad in sorting the documents by relevance.

    With a BiEncoderRanker, you can:
    - directly get predictions via predict()

    Usage example:

    ```python
    retriever = BM25Retriever(document_store=document_store)
    ranker = BiEncoderRanker(model_name_or_path="sentence-transformers/all-MiniLM-L12-v2")
    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        model_version: Optional[str] = None,
        top_k: int = 10,
        use_gpu: bool = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        batch_size: int = 16,
        scale_score: bool = True,
        pooling: str = "mean",  # "mean" and "cls" are supported
        progress_bar: bool = True,
        query_prompt: Optional[str] = None,
        document_prompt: Optional[str] = None,
        pad_to_max: Optional[bool] = False,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'sentence-transformers/all-MiniLM-L12-v2'
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param top_k: The maximum number of documents to return
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to process at a time.
        :param scale_score: The raw predictions will be transformed using a Sigmoid activation function in case the model
                    only predicts a single label. For multi-label predictions, no scaling is applied. Set this
                    to False if you do not want any scaling of the raw predictions.
        :param progress_bar: Whether to show a progress bar while processing the documents.
        :param use_auth_token: The API token used to download private models from Huggingface.
                        If this parameter is set to `True`, then the token generated when running
                        `transformers-cli login` (stored in ~/.huggingface) will be used.
                        Additional information can be found here
                        https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                A list containing torch device objects and/or strings is supported (For example
                [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                parameter is not used and a single cpu device is used for inference.
        :param pooling: The pooling method to use to represent the encoded document or query. Supported methods: 'cls' - take the [CLS] token vector; 'mean' - mean of all the document/query vectors.
        :param query_prompt: Optional. A prefix to add to a query representation.
        :param document_prompt: Optional. A prefix to add to a document representation.
        :param pad_to_max: Optional. Pad the samples to the model's maximum number of positions instead of per batch maximum number of tokens. Recommanded to use when embedding many documents. Works only in `predict_batch`.
        """
        torch_and_transformers_import.check()
        super().__init__()

        self.top_k = top_k

        self.devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=True
        )

        self.progress_bar = progress_bar
        self.model_name_or_path = model_name_or_path
        self.model_version = model_version
        self.use_auth_token = use_auth_token
        self.batch_size = batch_size
        self.normalize = scale_score
        self.pooling = pooling
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.pad_to_max = pad_to_max
        self.warm_up()

    def warm_up(self):
        self.transformer_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            revision=self.model_version,
            use_auth_token=self.use_auth_token,
        )
        self.transformer_model.to(str(self.devices[0]))
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            revision=self.model_version,
            use_auth_token=self.use_auth_token,
        )
        self.transformer_model.eval()

        if len(self.devices) > 1:
            self.transformer_model = DataParallel(self.transformer_model, device_ids=self.devices)

    def _embed(self, inputs):
        with torch.inference_mode():
            outputs = self.transformer_model(**inputs)
            if self.pooling == "mean":
                emb = self._mean_pooling(outputs, inputs["attention_mask"])
            elif self.pooling == "cls":
                emb = self._cls_pooling(outputs)
            else:
                raise ValueError("pooling method no supported")

            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb

    @staticmethod
    def _cls_pooling(outputs):
        if type(outputs) == dict:
            token_embeddings = outputs["last_hidden_state"]
        else:
            token_embeddings = outputs[0]
        return token_embeddings[:, 0]

    @staticmethod
    def _mean_pooling(outputs, attention_mask):
        if type(outputs) == dict:
            token_embeddings = outputs["last_hidden_state"]
        else:
            # First element of model_output contains all token embeddings
            token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def predict(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Use loaded bi-encoder ranker model to re-rank the supplied list of Document.

        Returns list of Document sorted by (desc.) similarity with the query.

        :param query: Query string
        :param documents: List of Document to be re-ranked
        :param top_k: The maximum number of documents to return
        :return: List of Document
        """
        if top_k is None:
            top_k = self.top_k

        if self.query_prompt:
            query = self.query_prompt + query
        query_tok = self.transformer_tokenizer(
            query, padding=True, truncation=True, return_tensors="pt"
        ).to(self.devices[0])
        docs = [
            d.content if not self.document_prompt else self.document_prompt + d.content
            for d in documents
        ]
        documents_tok = self.transformer_tokenizer(
            docs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.devices[0])

        q_vec = self._embed(query_tok)
        docs_vec = self._embed(documents_tok)
        scores = q_vec @ docs_vec.T
        scores = scores.reshape(len(docs))
        indices = scores.cpu().sort(descending=True).indices
        return [documents[i.item()] for i in indices[:top_k]]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[Document]]:
        """
        Use loaded bi-encoder ranker model to re-rank the supplied lists of Documents.

        Returns lists of Documents sorted by (desc.) similarity with the corresponding queries.


        - If you provide a list containing a single query...

            - ... and a single list of Documents, the single list of Documents will be re-ranked based on the
              supplied query.
            - ... and a list of lists of Documents, each list of Documents will be re-ranked individually based on the
              supplied query.


        - If you provide a list of multiple queries...

            - ... you need to provide a list of lists of Documents. Each list of Documents will be re-ranked based on
              its corresponding query.

        :param queries: Single query string or list of queries
        :param documents: Single list of Documents or list of lists of Documents to be reranked.
        :param top_k: The maximum number of documents to return per Document list.
        :param batch_size: Number of Documents to process at a time.
        """
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        # encode all the queries, if >1 encode using batches
        # - if >1 only accept list of lists of docs
        # encode all documents:
        # - single list: just encode with batch size
        # - multi-list: concatenate all, encode with batch and re-order; single query or |queries| == lists

        query_batches = self._get_batches(queries, batch_size)
        pb = tqdm(total=len(queries), disable=not self.progress_bar, desc="Encoding queries")
        query_vectors = []
        for batch in query_batches:
            _q = [q if not self.query_prompt else self.query_prompt + q for q in batch]
            query_tok = self.transformer_tokenizer(
                _q, padding=True, truncation=True, return_tensors="pt"
            ).to(self.devices[0])
            q_vecs = self._embed(query_tok)
            query_vectors.extend(q_vecs)
            pb.update(len(batch))
        pb.close()
        query_vectors = torch.stack(query_vectors).reshape(len(queries), -1)

        (
            number_of_docs,
            all_qids,
            all_docs,
            single_list_of_docs,
        ) = self._preprocess_batch_queries_and_docs(queries=queries, documents=documents)

        doc_batches = self._get_document_batches(all_qids, all_docs, batch_size)
        pb = tqdm(total=len(queries), disable=not self.progress_bar, desc="Encoding documents")

        preds = []
        for qids, batch in doc_batches:
            _d = [
                d.content if not self.document_prompt else self.document_prompt + d.content
                for d in batch
            ]
            documents_tok = self.transformer_tokenizer(
                _d,
                padding="max_length" if self.pad_to_max else True,
                truncation=True,
                return_tensors="pt",
            ).to(self.devices[0])
            docs_vec = self._embed(documents_tok)
            scores = torch.bmm(
                query_vectors[qids].reshape(len(qids), 1, -1), docs_vec.reshape(len(qids), -1, 1)
            ).flatten()
            preds.extend(scores.detach().cpu().numpy())
            pb.update(len(batch))
        pb.close()

        if single_list_of_docs:
            sorted_scores_and_documents = sorted(
                zip(preds, documents),
                key=lambda similarity_document_tuple:
                # assume the last element in logits represents the `has_answer` label
                similarity_document_tuple[0],
                reverse=True,
            )

            # is this step needed?
            sorted_documents = [
                (score, doc)
                for score, doc in sorted_scores_and_documents
                if isinstance(doc, Document)
            ]

            return self._add_score_to_documents(sorted_documents[:top_k])
        else:
            # Group predictions together
            grouped_predictions = []
            left_idx = 0
            for number in number_of_docs:
                right_idx = left_idx + number
                grouped_predictions.append(preds[left_idx:right_idx])
                left_idx = right_idx

            result = []
            for pred_group, doc_group in zip(grouped_predictions, documents):
                sorted_scores_and_documents = sorted(
                    zip(pred_group, doc_group),  # type: ignore
                    key=lambda similarity_document_tuple:
                    # assume the last element in logits represents the `has_answer` label
                    similarity_document_tuple[0],
                    reverse=True,
                )

                # rank documents according to scores
                sorted_documents = [
                    (score, doc)
                    for score, doc in sorted_scores_and_documents
                    if isinstance(doc, Document)
                ]

                result.append(self._add_score_to_documents(sorted_documents[:top_k]))

            return result

    def _add_score_to_documents(
        self, sorted_scores_and_documents: List[Tuple[Any, Document]]
    ) -> List[Document]:
        def set_scores(x):
            x[1].score = x[0]
            return x[1]

        return list(map(set_scores, sorted_scores_and_documents))

    def _preprocess_batch_queries_and_docs(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]]
    ) -> Tuple[List[int], List[str], List[Document], bool]:
        number_of_docs = []
        all_qids = []
        all_docs: List[Document] = []
        single_list_of_docs = False

        # Docs case 1: single list of Documents -> rerank single list of Documents based on single query
        if len(documents) > 0 and isinstance(documents[0], Document):
            if len(queries) != 1:
                raise HaystackError(
                    "Number of queries must be 1 if a single list of Documents is provided."
                )
            query = queries[0]
            number_of_docs = [len(documents)]
            all_qids = [0] * len(documents)
            all_docs = documents  # type: ignore
            single_list_of_docs = True

        # Docs case 2: list of lists of Documents -> rerank each list of Documents based on corresponding query
        # If queries contains a single query, apply it to each list of Documents
        if len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                qids = [0] * len(documents)
                queries = queries * len(documents)
            elif len(queries) != len(documents):
                raise HaystackError(
                    "Number of queries must be equal to number of provided Document lists."
                )
            else:
                qids = range(len(queries))
            for qid, cur_docs in zip(qids, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(
                        f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents."
                    )
                number_of_docs.append(len(cur_docs))
                all_qids.extend([qid] * len(cur_docs))
                all_docs.extend(cur_docs)

        return number_of_docs, all_qids, all_docs, single_list_of_docs

    @staticmethod
    def _get_document_batches(
        all_qids: List[int], all_docs: List[Document], batch_size: Optional[int]
    ) -> Iterator[Tuple[List[int], List[Document]]]:
        if batch_size is None:
            yield all_qids, all_docs
            return
        else:
            for index in range(0, len(all_qids), batch_size):
                yield all_qids[index : index + batch_size], all_docs[index : index + batch_size]

    @staticmethod
    def _get_batches(
        samples: Union[List[str], List[Document]], batch_size: Optional[int]
    ) -> Iterator[List]:
        if batch_size is None:
            yield samples
            return
        else:
            for i in range(0, len(samples), batch_size):
                yield samples[i : i + batch_size]
