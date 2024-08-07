import random
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from embedders import EmbedderModelMTEB
import mteb
from mteb import MTEB
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel import INCQuantizer, IPEXModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


def load_msmarco_calibration_set(sample_size) -> Dataset:
    corpus = load_dataset("BeIR/msmarco", "corpus")
    queries = load_dataset("BeIR/msmarco", "queries")

    random.seed(1911)
    random_queries = random.sample(range(len(queries["queries"])), sample_size // 2)
    random_docs = random.sample(range(len(corpus["corpus"])), sample_size // 2)

    random_queries = [queries["queries"][x]["text"] for x in random_queries]
    random_docs = [corpus["corpus"][x]["text"] for x in random_docs]
    samples = random_queries + random_docs
    random.shuffle(samples)

    def gen():
        for s in samples:
            yield {"text": s}

    return Dataset.from_generator(gen)


def load_qasper_calibration_set(sample_size) -> Dataset:
    dataset = load_dataset("allenai/qasper")
    train_set = dataset["train"]

    random.seed(666)
    random_samples = random.sample(range(len(train_set)), sample_size)

    random_queries = [random.sample(train_set[x]["qas"]["question"], 1)[0] for x in random_samples]
    random_abstracts = [train_set[x]["abstract"] for x in random_samples]

    samples = random.sample(random_queries + random_abstracts, sample_size)
    random.shuffle(samples)

    def gen():
        for s in samples:
            yield {"text": s}

    return Dataset.from_generator(gen)


def load_rerank_calibration_set(sample_size) -> Dataset:
    mind = load_dataset("mteb/mind_small")
    scidocs = load_dataset("mteb/scidocs-reranking")
    stackdocs = load_dataset("mteb/stackoverflowdupquestions-reranking")

    rerank_test_dist = {
        "stackdocs": stackdocs["train"],
        "mind": mind["train"],
        "scidocs": scidocs["validation"],
    }

    total = 0
    for k, v in rerank_test_dist.items():
        size = len(v["query"]) + len(v["positive"]) + len(v["negative"])
        total += size
        print(f"dataset: {k} ; size: {size}")
    print(f"total samples loaded: {total}")

    all_samples = []
    for d in rerank_test_dist.values():
        for sample in d:
            if type(sample["query"]) == list:
                all_samples.extend(sample["query"])
            else:
                all_samples.append(sample["query"])
            all_samples.extend(sample["positive"])
            all_samples.extend(sample["negative"])

    assert len(all_samples) >= sample_size, "requested more than available samples"

    random.seed(1911)
    random_sample = random.sample(all_samples, sample_size)
    random.shuffle(random_sample)

    def gen():
        for s in random_sample:
            yield {"text": s}

    return Dataset.from_generator(gen)


def quantize(model_name, output_path, calibration_set):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

    vectorized_ds = calibration_set.map(preprocess_function, num_proc=10)
    vectorized_ds = vectorized_ds.remove_columns(["text"])

    quantizer = INCQuantizer.from_pretrained(model)
    quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", domain="nlp")
    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=vectorized_ds,
        save_directory=output_path,
        batch_size=1,
    )
    tokenizer.save_pretrained(output_path)


benchmarks = {
    "rerank": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "retrieval": [
        "ArguAna",
        "ClimateFEVER",
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "DBPedia",
        "FaithDial",
        "FeedbackQARetrieval",
        "FEVER",
        "FiQA2018",
        "HagridRetrieval",
        "HotpotQA",
        "LegalBenchConsumerContractsQA",
        "LegalBenchCorporateLobbying",
        "LegalSummarization",
        "MLQuestions",
        "MSMARCO",
        "NarrativeQARetrieval",
        "NFCorpus",
        "NQ",
        "RARbCode",
        "RARbMath",
        "SCIDOCS",
        "SciFact",
        "TopiOCQA",
        "Touche2020",
        "TRECCOVID",
    ],
}


def _gather_rerank_results(results):
    res = {}
    total = 0.0
    for task in results:
        res[task.task_name] = task.scores["test"][0]["map"]
        total += res[task.task_name]
    res["avg"] = total / len(results)
    return res


def _gather_retrieval_results(results):
    res = {}
    total = 0.0
    for task in results:
        res[task.task_name] = task.scores["test"][0]["ndcg_at_10"]
        total += res[task]
    res["avg"] = total / len(results)
    return res


results_fn = {
    "rerank": _gather_rerank_results,
    "retrieval": _gather_retrieval_results,
}

def _run_validation(model, task, model_path):
    evaluation = MTEB(tasks=mteb.get_tasks(tasks=benchmarks[task]))
    results = evaluation.run(
        model, overwrite_results=True, output_folder=model_path, eval_splits=["test"]
    )
    acc = results_fn[task](results)
    print("Detailed results")
    print(f"model: {model_path}")
    print("--------------------------")
    print(acc)
    return acc


@torch.inference_mode()
def validate_model(model_path, task, pooling, is_optimized=False):
    qm = IPEXModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if is_optimized:
        query_prompt = None
        if task == "retrieval":
            query_prompt = "Represent this sentence for searching relevant passages: "
        model = EmbedderModelMTEB(qm, tokenizer, pooling=pooling, query_prompt=query_prompt)
    else:
        model = SentenceTransformer(model_path)
    _run_validation(model, task, model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--task", type=str, default="rerank")
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--use_cls", action="store_true")

    args = parser.parse_args()

    assert (args.quantize and not args.benchmark) or (
        not args.quantize and args.benchmark
    ), "quantization and benchmark cannot be selected together"
    if args.quantize:
        # dataset = load_rerank_calibration_set(args.sample_size)
        # dataset = load_msmarco_calibration_set(args.sample_size)
        dataset = load_qasper_calibration_set(args.sample_size)
        quantize(args.model_name, args.output_path, calibration_set=dataset)
    if args.benchmark:
        validate_model(
            model_path=args.model_name,
            task=args.task,
            pooling="cls" if args.use_cls else "mean",
            is_optimized=args.opt,
        )
