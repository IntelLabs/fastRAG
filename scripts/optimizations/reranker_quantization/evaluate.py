"""
This file evaluates CrossEncoder on the TREC 2019 Deep Learning (DL) Track: https://arxiv.org/abs/2003.07820

TREC 2019 DL is based on the corpus of MS Marco. MS Marco provides a sparse annotation, i.e., usually only a single
passage is marked as relevant for a given query. Many other highly relevant passages are not annotated and hence are treated
as an error if a model ranks those high.

TREC DL instead annotated up to 200 passages per query for their relevance to a given query. It is better suited to estimate
the model performance for the task of reranking in Information Retrieval.

Run:
python eval_cross-encoder-trec-dl.py cross-encoder-model-name

"""
import argparse
import gzip
import logging
import os
import time
from argparse import Namespace
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import pytrec_eval
import torch
import tqdm
from sentence_transformers import util
from transformers import AutoTokenizer


def smart_batching_collate_onnx(batch, max_length, tensor_list_type="graph"):
    texts = [[] for _ in range(len(batch[0].texts))]

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text.strip())

    tokenized = tokenizer(
        *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=max_length
    )

    if tensor_list_type == "graph":
        tensor_list = [
            tokenized["input_ids"].numpy().astype("int32"),
            tokenized["token_type_ids"].numpy().astype("int32"),
            tokenized["attention_mask"].numpy().astype("int32"),
        ]

    if tensor_list_type == "ort":
        tensor_list = {
            "input_ids": tokenized["input_ids"].numpy().astype("int64"),
            "attention_mask": tokenized["attention_mask"].numpy().astype("int64"),
            "token_type_ids": tokenized["token_type_ids"].numpy().astype("int64"),
        }

    return tensor_list


def smart_batching_collate(batch, max_length):
    texts = [[] for _ in range(len(batch[0].texts))]

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text.strip())

    tokenized = tokenizer(
        *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=max_length
    )

    return dict(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        token_type_ids=tokenized["token_type_ids"],
    )


class BaseRunner:
    def __init__(self, model_path, max_length, batch_size, device):
        self.max_length = max_length
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.load_model(model_path)

    def load_model(self, model_path):
        pass

    def predict(self, query, corpus_sentences):
        pass

    def simple_batch_size_division(self, batch):
        bsz = self.batch_size

        tensor_size = self.get_tensor_size(batch)
        chunk_count = tensor_size // bsz
        if chunk_count != tensor_size / bsz:
            chunk_count += 1

        pred = []
        for bsz_i in tqdm.tqdm(list(range(chunk_count)), desc="Batches: "):
            sub_batch = self.get_sub_batch(batch, bsz, bsz_i)

            sub_pred = self.get_predictions(sub_batch)
            pred += sub_pred

        return pred


class CrossEncoderRunner(BaseRunner):
    def __init__(self, model_path, max_length, batch_size, device):
        super().__init__(model_path, max_length, batch_size, device)

    def load_model(self, model_path):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_path, max_length=self.max_length, device=device)

    def predict(self, query, corpus_sentences):
        cross_inp = [[query, sent] for sent in corpus_sentences]

        start_time = time.monotonic()

        if (
            self.model.config.num_labels > 1
        ):  # Cross-Encoder that predict more than 1 score, we use the last and apply softmax
            cross_scores = self.model.predict(
                cross_inp, apply_softmax=True, batch_size=self.batch_size
            )[:, 1].tolist()
        else:
            cross_scores = self.model.predict(cross_inp, batch_size=self.batch_size).tolist()

        end_time = time.monotonic()
        delta = timedelta(seconds=end_time - start_time)

        return cross_scores, delta


class HFRunner(BaseRunner):
    def __init__(self, model_path, max_length, batch_size, device):
        super().__init__(model_path, max_length, batch_size, device)

    def load_model(self, model_path):
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, query, corpus_sentences):
        cross_inp = [Namespace(texts=[query, sent]) for sent in corpus_sentences]
        batch = smart_batching_collate(cross_inp, self.max_length)

        start_time = time.monotonic()

        with torch.no_grad():
            pred = self.simple_batch_size_division(batch)

        end_time = time.monotonic()
        delta = timedelta(seconds=end_time - start_time)

        return pred, delta

    def get_tensor_size(self, batch):
        return batch["input_ids"].shape[0]

    def get_sub_batch(self, batch, bsz, bsz_i):
        return {k: batch[k][bsz * bsz_i : bsz * (bsz_i + 1)] for k in batch}

    def get_predictions(self, batch):
        result = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        pred = result.logits.reshape(-1).tolist()
        return pred


class OptimizedRunner(HFRunner):
    def __init__(self, model_path, max_length, batch_size, device):
        super().__init__(model_path, max_length, batch_size, device)

    def load_model(self, model_path):
        from intel_extension_for_transformers.optimization import OptimizedModel
        from neural_compressor.model.torch_model import PyTorchFXModel

        graph = OptimizedModel.from_pretrained(model_path)
        self.model = PyTorchFXModel(graph)
        self.model.eval()


class GraphRunner(BaseRunner):
    def __init__(self, model_path, max_length, batch_size, device):
        super().__init__(model_path, max_length, batch_size, device)

    def load_model(self, model_path):
        from intel_extension_for_transformers.backends.neural_engine.compile import compile

        self.model = compile(model_path)

    def predict(self, query, corpus_sentences):
        cross_inp = [Namespace(texts=[query, sent]) for sent in corpus_sentences]
        batch = smart_batching_collate_onnx(cross_inp, self.max_length)

        start_time = time.monotonic()

        pred = self.simple_batch_size_division(batch)

        end_time = time.monotonic()
        delta = timedelta(seconds=end_time - start_time)

        return pred, delta

    def get_tensor_size(self, batch):
        return batch[0].shape[0]

    def get_sub_batch(self, batch, bsz, bsz_i):
        sub_batch = [m[bsz * bsz_i : bsz * (bsz_i + 1)] for m in batch]
        return sub_batch

    def get_predictions(self, batch):
        predictions = self.model.inference(batch)

        pred_key = list(predictions.keys())[0]
        preds = predictions[pred_key]

        cross_scores = preds.reshape(-1).tolist()
        return cross_scores


class ORTRunner(BaseRunner):
    def __init__(self, model_path, max_length, batch_size, device):
        super().__init__(model_path, max_length, batch_size, device)

    def load_model(self, model_path):
        import onnx
        import onnxruntime as ort

        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(self.model.SerializeToString(), None)

    def predict(self, query, corpus_sentences):
        cross_inp = [Namespace(texts=[query, sent]) for sent in corpus_sentences]
        batch = smart_batching_collate_onnx(cross_inp, self.max_length, tensor_list_type="ort")

        start_time = time.monotonic()

        pred = self.simple_batch_size_division(batch)

        end_time = time.monotonic()
        delta = timedelta(seconds=end_time - start_time)

        return pred, delta

    def get_tensor_size(self, batch):
        return batch["input_ids"].shape[0]

    def get_sub_batch(self, batch, bsz, bsz_i):
        sub_batch = {k: m[bsz * bsz_i : bsz * (bsz_i + 1)] for k, m in batch.items()}
        return sub_batch

    def get_predictions(self, batch):
        cross_scores = self.session.run(None, batch)
        cross_scores = cross_scores[0].reshape(-1).tolist()

        return cross_scores


MODEL_TYPE_RUNNER_MAP = {
    "hf": HFRunner,
    "optimized": OptimizedRunner,
    "onnx_graph": GraphRunner,
    "cross_encoder": CrossEncoderRunner,
    "ort": ORTRunner,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the TREC DL 2019 dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model or onnx file.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODEL_TYPE_RUNNER_MAP.keys()),
        help="""
        The model type to evaluate.
        hf: Regular Huggingface model, loaded with from_pretrained.
        optimized: An Optimized model, loaded with the OptimizedModel class.
        onnx_graph: An onnx file of the sparse and quantized model. WARNING: must run on a machine with a CPU that supports AVX512 or VNNI ISA.
        cross_encoder: A huggingface model that will be loaded using the CrossEncoder class, provided by the SBERT library.
    """,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer. If not provided, using the model_path instead.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for the concatenation of the question and the passage.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size fed into the model during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use, with the default being cpu.",
    )
    parser.add_argument(
        "--queries_file_url",
        type=str,
        default=None,
        help="A URL from which to download the queries file msmarco-test2019-queries.tsv.gz, in case it is not stored in the data_folder.",
    )
    parser.add_argument(
        "--qrels_file_url",
        type=str,
        default=None,
        help="A URL from which to download the query relevance file 2019qrels-pass.txt, in case it is not stored in the data_folder.",
    )
    parser.add_argument(
        "--top_passages_file_url",
        type=str,
        default=None,
        help="A URL from which to download the top 1000 pssages file msmarco-passagetest2019-top1000.tsv.gz, in case it is not stored in the data_folder.",
    )

    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path if args.tokenizer_path is not None else args.model_path

    runner = MODEL_TYPE_RUNNER_MAP[args.model_type](
        args.model_path, args.max_length, args.batch_size, args.device
    )

    data_folder = "trec2019-data"
    os.makedirs(data_folder, exist_ok=True)

    # Read test queries
    queries = {}
    queries_filepath = os.path.join(data_folder, "msmarco-test2019-queries.tsv.gz")
    if not os.path.exists(queries_filepath):
        logging.info("Download " + os.path.basename(queries_filepath))
        util.http_get(
            args.queries_file_url,
            queries_filepath,
        )

    with gzip.open(queries_filepath, "rt", encoding="utf8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    # Read which passages are relevant
    relevant_docs = defaultdict(lambda: defaultdict(int))
    qrels_filepath = os.path.join(data_folder, "2019qrels-pass.txt")

    if not os.path.exists(qrels_filepath):
        logging.info("Download " + os.path.basename(qrels_filepath))
        util.http_get(args.qrels_file_url, qrels_filepath)

    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, score = line.strip().split()
            score = int(score)
            if score > 0:
                relevant_docs[qid][pid] = score

    # Only use queries that have at least one relevant passage
    relevant_qid = []
    for qid in queries:
        if len(relevant_docs[qid]) > 0:
            relevant_qid.append(qid)

    # Read the top 1000 passages that are supposed to be re-ranked
    passage_filepath = os.path.join(data_folder, "msmarco-passagetest2019-top1000.tsv.gz")

    if not os.path.exists(passage_filepath):
        logging.info("Download " + os.path.basename(passage_filepath))
        util.http_get(
            args.top_passages_file_url,
            passage_filepath,
        )

    passage_cand = {}
    with gzip.open(passage_filepath, "rt", encoding="utf8") as fIn:
        for line in fIn:
            qid, pid, query, passage = line.strip().split("\t")
            if qid not in passage_cand:
                passage_cand[qid] = []

            passage_cand[qid].append([pid, passage])

    logging.info("Queries: {}".format(len(queries)))

    queries_result_list = []
    run = {}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    run_metrics = []
    for qid in tqdm.tqdm(relevant_qid):
        query = queries[qid]

        cand = passage_cand[qid]
        pids = [c[0] for c in cand]
        corpus_sentences = [c[1] for c in cand]
        print(f"Corpus Sentence Count: {len(corpus_sentences)}")

        cross_scores, latency_delta = runner.predict(query, corpus_sentences)

        run_metrics.append(latency_delta)

        cross_scores_sparse = {}
        for idx, pid in enumerate(pids):
            cross_scores_sparse[pid] = cross_scores[idx]

        sparse_scores = cross_scores_sparse
        run[qid] = {}
        for pid in sparse_scores:
            run[qid][pid] = float(sparse_scores[pid])

    print(f"latency: {pd.DataFrame(run_metrics).mean()}")
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {"ndcg_cut.10"})
    scores = evaluator.evaluate(run)

    print("Queries:", len(relevant_qid))
    print("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores.values()]) * 100))
