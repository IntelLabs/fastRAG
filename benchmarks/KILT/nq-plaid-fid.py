import logging
import os
import pathlib

import kilt.eval_retrieval as retrieval_metrics
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader, PromptModel, SentenceTransformersRanker
from haystack.nodes.prompt import AnswerParser, PromptNode
from haystack.nodes.prompt.prompt_template import PromptTemplate
from kilt.eval_downstream import _calculate_metrics, validate_input
from tqdm import tqdm

from fastrag.prompters.invocation_layers import fid
from fastrag.retrievers.colbert import ColBERTRetriever
from fastrag.stores import PLAIDDocumentStore
from fastrag.utils import get_timing_from_pipeline


def evaluate(gold_records, guess_records):
    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)

    # 2. retrieval performance
    retrieval_results = retrieval_metrics.compute(
        gold_records, guess_records, ks=[1, 5], rank_keys=["wikipedia_id"]
    )
    result["retrieval"] = {
        "Rprec": retrieval_results["Rprec"],
        "recall@5": retrieval_results["recall@5"],
    }

    return result


def create_json_entry(jid, input_text, answer, documents):
    return {
        "id": jid,
        "input": input_text,
        "output": [{"answer": answer, "provenance": [{"wikipedia_id": d.id} for d in documents]}],
    }


def create_records(test_dataset, result_collection):
    guess_records = []

    for i in range(len(test_dataset)):
        example = test_dataset[i]
        results = result_collection[i]
        guess_records.append(
            create_json_entry(
                example["id"], example["input"], results["answers"][0].answer, results["documents"]
            )
        )
    return guess_records


def evaluate_from_answers(gold_records, result_collection):
    guess_records = create_records(gold_records, result_collection)

    return evaluate(gold_records, guess_records)


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logging.info("Loading PLAID index...")

document_store = PLAIDDocumentStore(
    collection_path="collection_path", checkpoint_path="checkpoint_path", index_path="index_path"
)

# Create Components

retriever = ColBERTRetriever(document_store=document_store)

reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

PrompterModel = PromptModel(
    model_name_or_path="Intel/fid_flan_t5_base_nq",
    use_gpu=True,
    invocation_layer_class=fid.FiDHFLocalInvocationLayer,
    model_kwargs=dict(
        model_kwargs=dict(device_map={"": 0}, torch_dtype=torch.bfloat16, do_sample=False),
        generation_kwargs=dict(max_length=10),
    ),
)

reader = PromptNode(
    model_name_or_path=PrompterModel,
    default_prompt_template=PromptTemplate("{query}", output_parser=AnswerParser()),
)


# Build Pipeline

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
p.add_node(component=reader, name="Reader", inputs=["Reranker"])


# Load Dataset

data = load_dataset("kilt_tasks", "nq")

validation_data = data["validation"]

# Run Pipeline

retriever_top_k = 100
reranker_top_k = 50


all_results = []
efficiency_metrics = []
for example in tqdm(validation_data):
    results = p.run(
        query=example["input"],
        params={"Retriever": {"top_k": retriever_top_k}, "Reranker": {"top_k": reranker_top_k}},
    )
    pipeline_latency_report = get_timing_from_pipeline(p)
    efficiency_metrics.append(
        {
            component_name: component_time[1]
            for component_name, component_time in pipeline_latency_report.items()
        }
    )
    all_results.append(results)

kilt_metrics = evaluate_from_answers(validation_data, all_results)


# Show Results
efficiency_metrics_df = pd.DataFrame(efficiency_metrics)
efficiency_metrics_df_mean = efficiency_metrics_df.mean()

for metric in efficiency_metrics_df.columns:
    logging.info(f"Mean Latency for {metric} examples: {efficiency_metrics_df_mean[metric]} sec")

logging.info(
    f"""
Accuracy: {kilt_metrics['downstream']['accuracy']}
EM: {kilt_metrics['downstream']['em']}
F1: {kilt_metrics['downstream']['f1']}
ROUGE-L: {kilt_metrics['downstream']['rougel']}
"""
)
