"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

In this example we use a knowledge distillation setup. Sebastian Hofstätter et al. trained in https://arxiv.org/abs/2010.02666
an ensemble of large Transformer models for the MS MARCO datasets and combines the scores from a BERT-base, BERT-large, and ALBERT-large model.

We use the logits scores from the ensemble to train a smaller model. We found that the MiniLM model gives the best performance while
offering the highest speed.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder-v2.py
"""
import argparse
import gzip
import logging
import os
import random
import shutil
import tarfile
from datetime import datetime

import model_compression_research as mcr
import torch
from sentence_transformers import InputExample, LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a sparse model on ")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Intel/bert-base-uncased-sparse-85-unstructured-pruneofa",
        help="Name of the model.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="The batch size for training"
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="",
        help="The path to save the model to. Default is the current path.",
    )
    parser.add_argument(
        "--data_folder", type=str, default="data", help="Path containing the training data."
    )
    parser.add_argument(
        "--num_dev_queries",
        type=int,
        default=200,
        help="Number of queries for the development set.",
    )
    parser.add_argument(
        "--num_max_dev_negatives",
        type=int,
        default=200,
        help="Maximum amount of negative examples.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5000, help="Number of warmup steps for training."
    )
    parser.add_argument(
        "--evaluation_steps", type=int, default=5000, help="Number of evaluation steps."
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate value.")
    parser.add_argument(
        "--skip_pruning",
        action="store_true",
        help="Whether to do the pruning for sparsity or not. Default is to perform pruning.",
    )
    parser.add_argument(
        "--collection_file_url",
        type=str,
        default=None,
        help="A URL from which to download the collection file collection.tar.gz, in case it is not stored in the data_folder.",
    )
    parser.add_argument(
        "--queries_file_url",
        type=str,
        default=None,
        help="A URL from which to download the queries file queries.tar.gz, in case it is not stored in the data_folder.",
    )
    parser.add_argument(
        "--triplet_file_url",
        type=str,
        default=None,
        help="A URL from which to download the triplet data file msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz, in case it is not stored in the data_folder.",
    )
    parser.add_argument(
        "--train_score_file_url",
        type=str,
        default=None,
        help="A URL from which to download the train score file bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv, in case it is not stored in the data_folder.",
    )

    args = parser.parse_args()

    fid = None
    model_name = args.model_name
    train_batch_size = args.train_batch_size
    num_epochs = args.num_epochs
    model_save_path = (
        args.model_save_path
        + model_name.replace("/", "-")
        + "-"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    data_folder = args.data_folder

    while fid is None or os.path.exists(fid):
        fid = os.path.abspath(__file__) + str(random.randint(0, 2**31 - 1))
    shutil.copy(os.path.abspath(__file__), fid)
    # First, we define the transformer model we want to fine-tune

    if not args.skip_pruning:
        pruning_config = mcr.OneShotPruningConfig(
            pruning_fn="pattern_lock", not_to_prune=["classifier"]
        )

    # We set num_labels=1 and set the activation function to Identiy, so that we get the raw logits
    model = CrossEncoder(
        model_name, num_labels=1, max_length=256, default_activation_function=torch.nn.Identity()
    )
    if not args.skip_pruning:
        pruning_scheduler = mcr.OneShotPruningScheduler(model.model, pruning_config)

    ### Now we read the MS Marco dataset
    os.makedirs(data_folder, exist_ok=True)

    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}
    collection_filepath = os.path.join(data_folder, "collection.tsv")
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, "collection.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get(
                args.collection_file_url,
                tar_filepath,
            )

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(collection_filepath, "r", encoding="utf8") as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage

    ### Read the train queries, store in queries dict
    queries = {}
    queries_filepath = os.path.join(data_folder, "queries.train.tsv")
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, "queries.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get(args.queries_file_url, tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(queries_filepath, "r", encoding="utf8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    ### Now we create our  dev data
    train_samples = []
    dev_samples = {}

    # We use 200 random queries from the train set for evaluation during training
    # Each query has at least one relevant and up to 200 irrelevant (negative) passages
    num_dev_queries = args.num_dev_queries
    num_max_dev_negatives = args.num_max_dev_negatives

    # msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
    # shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
    # We extracted in the train-eval split 500 random queries that can be used for evaluation during training
    train_eval_filepath = os.path.join(
        data_folder, "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz"
    )
    if not os.path.exists(train_eval_filepath):
        logging.info("Download " + os.path.basename(train_eval_filepath))
        util.http_get(
            args.triplet_file_url,
            train_eval_filepath,
        )

    with gzip.open(train_eval_filepath, "rt") as fIn:
        for line in fIn:
            qid, pos_id, neg_id = line.strip().split()

            if qid not in dev_samples and len(dev_samples) < num_dev_queries:
                dev_samples[qid] = {"query": queries[qid], "positive": set(), "negative": set()}

            if qid in dev_samples:
                dev_samples[qid]["positive"].add(corpus[pos_id])

                if len(dev_samples[qid]["negative"]) < num_max_dev_negatives:
                    dev_samples[qid]["negative"].add(corpus[neg_id])

    dev_qids = set(dev_samples.keys())

    # Read our training file
    # As input examples, we provide the (query, passage) pair together with the logits score from the teacher ensemble
    teacher_logits_filepath = os.path.join(
        data_folder, "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
    )
    train_samples = []
    if not os.path.exists(teacher_logits_filepath):
        util.http_get(
            args.train_score_file_url,
            teacher_logits_filepath,
        )

    with open(teacher_logits_filepath) as fIn:
        for line in fIn:
            pos_score, neg_score, qid, pid1, pid2 = line.strip().split("\t")

            if qid in dev_qids:  # Skip queries in our dev dataset
                continue

            train_samples.append(
                InputExample(texts=[queries[qid], corpus[pid1]], label=float(pos_score))
            )
            train_samples.append(
                InputExample(texts=[queries[qid], corpus[pid2]], label=float(neg_score))
            )

    # We create a DataLoader to load our train samples
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True
    )

    # We add an evaluator, which evaluates the performance during training
    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

    # Configure the training
    warmup_steps = args.warmup_steps
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=torch.nn.MSELoss(),
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=args.evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        optimizer_params={"lr": args.lr},
        use_amp=True,
    )

    if not args.skip_pruning:
        pruning_scheduler.remove_pruning()
    # Save latest model
    output_dir = model_save_path + "-latest"
    model.save(output_dir)
    shutil.copy(fid, output_dir)
