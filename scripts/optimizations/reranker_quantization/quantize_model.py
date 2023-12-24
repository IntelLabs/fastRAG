import argparse
import json
from argparse import Namespace
from functools import partial

import numpy as np
import pandas as pd
import torch
from intel_extension_for_transformers.optimization.config import QuantizationConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from intel_extension_for_transformers.optimization.utils import metrics, objectives
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
from sentence_transformers import CrossEncoder
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments


def smart_batching_collate_wrapper(batch, collate_fn):
    tokenized, labels = collate_fn(batch)
    return dict(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        token_type_ids=tokenized["token_type_ids"],
        labels=labels,
    )


def create_rerank_examples(example, data_index_label, is_eval, ctx_count):
    data_index_label += 1
    answer = example["answers"][0]
    contexts = example["ctxs"][:ctx_count]
    neg_label = 0 if not is_eval else -1 * data_index_label
    pos_label = 1 if not is_eval else data_index_label

    for cc in contexts:
        assert answer != cc["text"]

    minibatch = [
        Namespace(texts=[example["question"], c["text"]], label=neg_label) for c in contexts
    ] + [Namespace(texts=[example["question"], answer], label=pos_label)]
    return minibatch


def evaluate(model):
    results = []
    print(f"eval_set: {len(eval_set)}")
    for example in tqdm(eval_set):
        contexts = example["ctxs"][: args.context_count]
        pairs = [[example["question"], c["text"]] for c in contexts] + [
            [example["question"], example["answers"][0]]
        ]

        batch = cross_encoder_model.smart_batching_collate_text_only(pairs)

        pred = model(**batch)

        gt = [0 for c in contexts] + [1]
        ap = average_precision_score(y_true=gt, y_score=pred.logits.reshape(-1).tolist())
        results.append({"ap": ap})
    return float(pd.DataFrame(results).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a sparse model on ")
    parser.add_argument("--model_path", type=str, help="Path to the model or onnx file.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="""
    Path to the data to calibrate and evaluate the model for quantization.
    The data file should be an array of json objects, with the following template:

    [
        {
            "question": "some question",
            "answers": [
                "some POSITIVE context"
            ],
            "ctxs": [
                {
                    "text": "NEGATIVE context #1
                },
                {
                    "text": "NEGATIVE context #2
                },
                ...
            ]
        }
    ]
    """,
    )
    parser.add_argument(
        "--train_instance_count",
        type=int,
        default=2000,
        help="Number of training (a.k.a calibration) examples.",
    )
    parser.add_argument(
        "--dev_instance_count", type=int, default=200, help="Number of evaluation examples."
    )
    parser.add_argument(
        "--context_count",
        type=int,
        default=30,
        help="Number of NEGATIVE contexts to provide per question, during evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp_quant_model",
        help="Directory to save the output quantized model into.",
    )

    args = parser.parse_args()

    train_instance_count = args.train_instance_count
    dev_instance_count = args.dev_instance_count

    model_path = args.model_path
    data_path = args.data_path

    cross_encoder_model = CrossEncoder(model_path)
    model_org = cross_encoder_model.model

    with open(data_path, "r") as f:
        dev_rank_data = json.load(f)

    calib_set = dev_rank_data[:train_instance_count]

    eval_set = dev_rank_data[train_instance_count : train_instance_count + dev_instance_count]
    assert calib_set[0]["ctxs"][0] != eval_set[0]["ctxs"][0]

    train_rank_data_objs = []
    for data_index_label, e in enumerate(tqdm(calib_set)):
        train_rank_data_objs += create_rerank_examples(e, data_index_label, False, 1)

    dev_rank_data_objs = []
    for data_index_label, e in enumerate(tqdm(eval_set)):
        dev_rank_data_objs += create_rerank_examples(e, data_index_label, False, 1)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        no_cuda=True,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
    )

    smart_batching_collate_wrapped = partial(
        smart_batching_collate_wrapper, collate_fn=cross_encoder_model.smart_batching_collate
    )

    trainer = NLPTrainer(
        model=model_org,
        args=train_args,
        data_collator=smart_batching_collate_wrapped,
        train_dataset=train_rank_data_objs,
        eval_dataset=dev_rank_data_objs,
    )

    trainer.args.dataloader_pin_memory = False

    try:
        import mlflow

        mlflow.end_run()
    except Exception as e:
        print(f"No mlflow installed.")

    metric = metrics.Metric(name="eval_ap", is_relative=True, criterion=0.0001)
    q_config = QuantizationConfig(
        approach="PostTrainingStatic",
        max_trials=200,  # set the Max tune times
        metrics=[metric],
        objectives=[objectives.performance],
    )

    # when you wish to quantie for fp23, you should set: trainer.enable_inc_quant = False
    #     trainer.enable_inc_quant = True
    trainer.enable_bf16 = False

    model = trainer.quantize(quant_config=q_config, eval_func=evaluate)

    trainer.enable_executor = True

    trainer.export_to_onnx()

    # save_for_huggingface_upstream(model, tokenizer, trainer.args.output_dir)
