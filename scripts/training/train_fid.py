import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import set_seed

from fastrag.readers.FiD import FusionInDecoderForConditionalGeneration, passages_to_tensors

try:
    from kilt.eval_downstream import _exact_match_score
except ImportError as ie:
    raise ImportError(
        "KILT was not found and is essential for calculating EM. Please install it via: pip install '.[benchmark]'"
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


INPUT_FILE_DESCRIPTION = """The file is a .json file, comprised of a list of json objects.
Each object follows this template:

{
    'id': 'Some id',
    'question': 'Some question?',
    'answers': ['Some answer...'],
    'ctxs': [
         {
             'id': 'Some id',
             'title': 'Some title',
             'text': 'Some text..'
         },
    ]
}
"""


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": f"The input training data file. {INPUT_FILE_DESCRIPTION}"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": f"The evaluation data. {INPUT_FILE_DESCRIPTION}"},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": f"The test data for predicting on questions and creating a submission file. {INPUT_FILE_DESCRIPTION}"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    passage_count: int = field(
        default=100,
        metadata={
            "help": "The amount of passages to use from the retrieved passage collection per question"
        },
    )
    train_with_random_answers: bool = field(
        default=False,
        metadata={
            "help": "If true, the a random answer will be drawn randomly every time an example is drawn during training."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


@dataclass
class FiDSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    metric_for_best_model: Optional[str] = field(
        default="eval_em", metadata={"help": "The metric to use to compare two different models."}
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."},
    )
    generation_max_length: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    compute_rouge_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to compute the ROUGE scores during evaluation."},
    )


def load_from_json_file(path):
    logging.info(f"Loading file {path}...")
    with open(path, "r") as f:
        data = json.load(f)
    return data


class FusionInDecoderDataset(Dataset):
    def __init__(
        self,
        examples,
        passage_count,
        answer_index=0,
        max_samples=-1,
        train=True,
        train_with_random_answers=False,
    ):
        self.passage_count = passage_count
        self.examples = examples if max_samples == -1 else examples[:max_samples]
        self.answer_index = answer_index

        self.pre_context = "context:"
        self.pre_title = "title:"
        self.pre_question = "question:"

        self.train = train
        self.train_with_random_answers = train_with_random_answers

    def get_example(self, idx):
        return self.examples[idx]

    def get_random_answer(self, answers):
        return random.choice(answers)

    def __getitem__(self, idx):
        example = self.examples[idx]

        documents = example["ctxs"][: self.passage_count]

        question = f"{self.pre_question} {example['question']}"

        formatted_passages = [
            f"{self.pre_title} {c['title']} {self.pre_context} {c['text']}" for c in documents
        ]

        formatted_passages_with_question = [question + " " + t for t in formatted_passages]

        if len(example["answers"]) > 0:
            if self.train_with_random_answers:
                answer = self.get_random_answer(example["answers"])
            else:
                answer = example["answers"][self.answer_index]
        else:
            answer = "No Answer"

        full_answer_list = example["answers"]

        item_dict = dict(
            question=question,
            formatted_passages_with_question=formatted_passages_with_question,
            answer=answer,
            full_answer_list=full_answer_list,
            idx=idx,
            train=self.train,
        )
        return item_dict

    def __len__(self):
        return len(self.examples)


class FusionInDecoderCollator:
    def __init__(self, tokenizer, passage_max_len=250, ans_max_len=50):
        self.tokenizer = tokenizer
        self.passage_max_len = passage_max_len
        self.ans_max_len = ans_max_len
        self.answer_mask_fill_value = -100

    def answer_to_tensor(self, answers):
        answer_tensor_obj = self.tokenizer.batch_encode_plus(
            answers,
            pad_to_max_length=True,
            max_length=self.ans_max_len,
            return_tensors="pt",
            truncation=True,
        )
        answer_tensor_obj_mask = answer_tensor_obj["attention_mask"].bool()
        labels = answer_tensor_obj["input_ids"].masked_fill(
            ~answer_tensor_obj_mask, self.answer_mask_fill_value
        )

        return labels

    def __call__(self, features, return_tensors=None):
        all_passages = [xitem["formatted_passages_with_question"] for xitem in features]
        all_input_ids, all_masks = passages_to_tensors(
            self.tokenizer, all_passages, self.passage_max_len, False
        )

        if features[0]["train"]:
            answer_texts = [xitem["answer"] for xitem in features]
            all_answers = self.answer_to_tensor(answer_texts)
        else:
            # in evaluation, use the idx to access the dataset, for the full text of the answers
            all_answers = torch.tensor([xitem["idx"] for xitem in features])
            all_answers = all_answers.view(-1, 1)

        input_tuple = dict(input_ids=all_input_ids, attention_mask=all_masks, labels=all_answers)
        return input_tuple


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FiDSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    current_seed = local_rank + training_args.seed
    logging.info(
        f"local_rank: {local_rank}, args.seed: {training_args.seed}, current_seed: {current_seed}, gradient_accumulation_steps: {training_args.gradient_accumulation_steps}, gradient_checkpointing: {training_args.gradient_checkpointing}"
    )
    set_seed(current_seed)

    # Get Data

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )

        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]
        test_dataset = raw_datasets["test"]

        if data_args.max_train_samples is not None:
            # We will select sample from whole data
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))
    else:
        assert (
            data_args.train_file is not None
            or data_args.validation_file is not None
            or data_args.test_file is not None
        ), "If a dataset was not specified, the train, dev or test files should be present."

        if data_args.train_file is not None:
            train_data = load_from_json_file(data_args.train_file)
            train_dataset = FusionInDecoderDataset(
                train_data,
                passage_count=data_args.passage_count,
                max_samples=data_args.max_train_samples,
                train=True,
                train_with_random_answers=data_args.train_with_random_answers,
            )
            logging.info(f"train_dataset length: {len(train_dataset)}")

        eval_dataset = None
        if data_args.validation_file is not None:
            dev_data = load_from_json_file(data_args.validation_file)
            eval_dataset = FusionInDecoderDataset(
                dev_data,
                passage_count=data_args.passage_count,
                max_samples=data_args.max_eval_samples,
                train=False,
            )
            logging.info(f"eval_dataset length: {len(eval_dataset)}")

        if data_args.test_file is not None:
            test_data = load_from_json_file(data_args.test_file)
            test_dataset = FusionInDecoderDataset(
                test_data,
                passage_count=data_args.passage_count,
                max_samples=data_args.max_predict_samples,
                train=False,
            )
            logging.info(f"test_dataset length: {len(test_dataset)}")

        # allow the collator to parse the json keys:
        training_args.remove_unused_columns = False

    # Create Model

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = FusionInDecoderForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define eval method

    collator = FusionInDecoderCollator(
        tokenizer, passage_max_len=data_args.max_seq_length, ans_max_len=data_args.max_answer_length
    )

    if training_args.compute_rouge_metrics:
        rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        if eval_dataset is None:
            return {}

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels_list = labels[:, 0].reshape(-1).tolist()
        eval_examples = [eval_dataset.__getitem__(i)["full_answer_list"] for i in labels_list]

        results_overall = []
        for current_labels_index in tqdm(
            list(range(len(eval_examples))), desc="Evaluating Examples"
        ):
            decoded_labels = eval_examples[current_labels_index]
            current_decoded_preds = decoded_preds[current_labels_index]

            current_results_overall = []
            for current_decoded_labels in decoded_labels:
                result = {}

                if training_args.compute_rouge_metrics:
                    result = rouge_metric.compute(
                        predictions=[current_decoded_preds],
                        references=[current_decoded_labels],
                        use_stemmer=True,
                    )
                    result = {k: round(v * 100, 4) for k, v in result.items()}

                result["em"] = _exact_match_score(current_decoded_preds, current_decoded_labels)

                current_results_overall.append(result)

            current_results_overall_dict = pd.DataFrame(current_results_overall).mean().to_dict()

            # if one of the answers matches, then EM is 1, else 0
            current_results_overall_dict["em"] = int(current_results_overall_dict["em"] > 0)
            results_overall.append(current_results_overall_dict)

        results_overall_dict = pd.DataFrame(results_overall).mean().to_dict()
        return results_overall_dict

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    if training_args.do_train:
        logging.info("*** Train ***")

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_len
    )
    num_beams = training_args.generation_num_beams
    if training_args.do_eval:
        logging.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval", max_length=max_length, num_beams=num_beams
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create a KILT submission file, with each line containing the answer and the input passages.
    if training_args.do_predict:
        logging.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.jsonl"
                )

                kilt_output = []
                for test_index in range(len(test_dataset)):
                    example = test_dataset.get_example(test_index)
                    predicted_value = predictions[test_index]

                    kilt_output.append(
                        {
                            "id": example["id"],
                            "input": example["question"],
                            "output": [
                                {
                                    "answer": predicted_value,
                                    "provenance": [
                                        {
                                            "wikipedia_id": test_context["id"],
                                            "wikipedia_title": test_context["title"],
                                        }
                                        for test_context in example["ctxs"]
                                    ],
                                }
                            ],
                        }
                    )

                with open(output_prediction_file, "w") as outfile:
                    for entry in kilt_output:
                        json.dump(entry, outfile)
                        outfile.write("\n")
