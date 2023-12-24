# Quantizing Ranking Models for Faster Inference

Quantization is a technique employed to reduce computational and memory costs during inference by utilizing low-precision data types, such as 8-bit integers (int8), in place of the conventional 32-bit floating point (float32) for representing weights and activations. This reduction in bit count leads to a model that requires less memory storage, potentially reduces energy consumption, and enables faster operations, such as matrix multiplication, through integer arithmetic.

We offer the capability to quantize models used in pipelines for Open-Domain tasks, with a specific focus on the cross encoder model. This model is responsible for reranking the retrieved documents.

In order to improve the efficiency of the cross encoder, we offer a variety of quantization and speed-up options. These options are specifically designed to expedite the inference process for cross encoders. Throughout this process, we employ the following techniques and tools:

* [Intel® Extension for Transformers: Accelerating Transformer-based Models on Intel Platforms](https://github.com/intel/intel-extension-for-transformers)
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

To generate the quantized model, we follow a series of steps:

1. Sparse Model Fine-Tuning: In this phase, we train a model that employs sparsity techniques. These techniques involve enforcing specific weights to be zero or close to zero, thereby reducing the overall number of parameters in the model. By training a sparse model, we can significantly decrease memory requirements and enhance the speed of inference.

2. Quantization: Entails representing the weights and activations of the model using data types with lower precision. For example, we can convert floating-point values to fixed-point or integer representations. This reduction in precision enables efficient storage and computation, resulting in faster inference times.

By following these steps of training a sparse model and subsequently quantizing it, we can achieve a quantized model that optimizes both memory efficiency and inference speed.

Once the model is prepared, it can be evaluated on downstream tasks to ensure that it maintains high performance in those specific tasks.

**IMPORTANT**: To ensure compatibility and avoid potential issues, it is crucial to follow the installation and quantization process outlined below step-by-step.

## Sparse Model Fine-Tuning

We present an example of training a sparse model on the MSMARCO dataset using the scripts provided in the [SBERT](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/ms_marco) library. For a more comprehensive understanding of the training process, we recommend referring to the detailed description available in the library. We note that the MS MARCO data is not meant for commercial use, and it used here for demonstrations purposes only.

To begin, we leverage the [Model Compression Research Package](https://github.com/IntelLabs/Model-Compression-Research-Package) to fine-tune a sparse model. This package offers a set of tools and techniques specifically designed for model compression purposes. We show an example of how to train a sparse model on the

Install the library using the following commands:

```bash
git clone https://github.com/IntelLabs/Model-Compression-Research-Package
cd Model-Compression-Research-Package
pip install -e .
```

The following script is designed to fine-tune a sparse model with **85%** sparsity, using the default settings provided in the script available [here](https://huggingface.co/Intel/bert-base-uncased-sparse-85-unstructured-pruneofa). The models supported for this script can be any encoder-based transformer model. To train the model without sparsity, you can include the `skip_pruning` parameter.

Execute the script using the following command:

```sh
python train_cross-encoder_kd.py --model_name MODEL_NAME --model_save_path SAVE_PATH --data_folder DATA_FOLDER
```

NOTE: If the `data_folder` you provide is empty, the script will download the MSMARCO dataset files, using these script arguments:

* **collection_file_url**: The URL for `collection.tar.gz`
* **queries_file_url**: The URL for `queries.tar.gz`
* **triplet_file_url**: The URL for `msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz`
* **train_score_file_url**: The URL for `bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv`

## Quantization

### Installing Required Packages

To successfully quantize the model, please ensure that you have the specific versions of the required packages listed in our setup.cfg file, under the "quantize_reranker" section.

To install these packages, execute the following command from the main directory:

```sh
pip install .[quantize_reranker]
```

NOTE: If you come across the error message `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, it indicates that there are missing package dependencies for the opencv package. To resolve this error, you can install opencv-python-headless instead. Execute the following command to install it:

```sh
bash fix_opencv_installation.sh
```

### Running the Quantization Process

In order to generate the quantized model, the first step is to calibrate it using a small dataset. To streamline this process, we have provided a script called [`quantize_model.py`](quantize_model.py) which handles the calibration and exports the model to the ONNX format.

Executing the script will produce a directory containing two essential files:

* The quantized HuggingFace model, saved in the standard format as `pytorch_model.bin` and `config.json`.
* An ONNX version of the model.

When calibrating the model, you will need to provide a calibration dataset. This dataset should be an array of JSON-formatted examples. Here's an example of how the calibration dataset should look like:

```json
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
```

To run the quantization process:

```sh
python quantize_model.py --model_path SPARSE_MODEL_PATH --data_path DATA_PATH
```

## Utilizing a Quantized Ranker in a fastRAG Pipeline

The quantized ranker model can be easily loaded into a `QuantizedRanker` node, allowing it to be used as part of a ranking pipeline. It can also serve as a ranker within a fastRAG pipeline.
Here's an example:

```python
from fastrag.rankers.quantized_reranker import QuantizedRanker
reranker = QuantizedRanker(
    onnx_file_path="ranker.onnx",
    tokenizer_path="<huggingface_tokenizer_name_or_path>"
)
```

As part of a pipeline:

```python
from haystack import Pipeline

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])

results = p.run(query="What is Paris?", params={"Retriever": {"top_k": 30}, "Reranker": {"top_k": 10}})
```

## Evaluation on a Downstream Task

To evaluate the performance of different model types, we provide a custom [evaluation script](evaluate.py) specifically designed for the TREC DL 2019 task. This script is sourced from the [SBERT site](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/eval_cross-encoder-trec-dl.py). The script allows you to evaluate multiple model types and measure the NDCG@10 metric, which aligns with the results published on the [MSMARCO Cross Encoder SBERT Results page](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html).

The `model_type` parameter is used to control the type of model you want to evaluate. Here are the available options:

* Standard Huggingface model: Set `model_type=hf` to evaluate this model type.
* Huggingface model with `CrossEncoder` class: Set `model_type=cross_encoder` to load the model using the `CrossEncoder` class provided by the SBERT library.
* Quantized model: Use `model_type=optimized` to evaluate a quantized model.
* ONNX graph model: Set `model_type=onnx_graph` to evaluate an ONNX graph model.

Please refer to the [evaluation script](evaluate.py) for further instructions on how to use it and interpret the results.

**IMPORTANT**: To effectively run the ONNX graph model, it is necessary to use a machine with a CPU that supports AVX512 or VNNI ISA.

To run the evaluation script:

```sh
python evaluate.py --model_path MODEL_PATH|ONNX_FILE_PATH --model_type onnx_graph|optimized|hf|cross_encoder --tokenizer_path TOKENIZER_PATH
```

NOTE: Please be aware that if the `data_folder` you provide is empty, the script will download the MSMARCO dataset into that folder, using the following script arguments:

* **queries_file_url**: The URL for `msmarco-test2019-queries.tsv.gz`
* **qrels_file_url**: The URL for `2019qrels-pass.txt`
* **top_passages_file_url**: The URL for `msmarco-passagetest2019-top1000.tsv.gz`
