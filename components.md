# fast**RAG** Components Overview



## REPLUG
<image align="right" src="assets/replug.png" width="600">

REPLUG (Retrieve and Plug) is a retrieval augmented LM method; documents are retrieved and then plugged in a prefix form
to the input; however this is done using ensembling, where the documents are processed in parallel and the final token
prediction is based on the combined probability distribution. This can be seen in the figure. REPLUG enables us to
process a larger number of retrieved documents, without limiting ourselves to the LLM context window. Additionally, this
method works with any LLM, no fine-tuning is needed. See ([Shi et al. 2023](#shiREPLUGRetrievalAugmentedBlackBox2023))
for more details.

We provide implementation for the REPLUG ensembling inference, using the invocation layer
`ReplugHFLocalInvocationLayer`; Our implementation supports most Hugging FAce models with `.generate()` capabilities (such that implement the generation mixin); For a complete example, see [REPLUG Parallel
Reader](examples/replug_parallel_reader.ipynb) notebook.

## ColBERT v2 with PLAID Engine

<image align="right" src="assets/colbert-maxsim.png" width="500">

ColBERT is a dense retriever which means it uses a neural network to encode all the documents into representative
vectors; once a query is made, it encodes the query into a vector and using vector similarity search, it finds the most
relevant documents for that query. What makes it different is the fact it stores the full vector representation of the
documents; neural network represent each word as a vector and previous models used a single vector to represent
documents, no matter how long they were. ColBERT stores the vectors of all the words in all the documents. It makes the
retrieving more accurate, albeit with the price of having a larger index. ColBERT v2 reduces the index size by
compressing the vectors using a technique called quantization. Finally, PLAID improves latency times for ColBERT-based
indexes by using a set of filtering steps, reducing the number of internal candidates to consider, reducing the amount
of computation needed for each query. Overall, ColBERT v2 with PLAID provide state of the art retrieving results with a
much smaller latency than previous dense retrievers, getting very close to sparse retrievers performance with much
higher accuracy. See ([Santhanam, Khattab, Saad-Falcon, et al. 2022](#org330b3f5); [Santhanam, Khattab, Potts, et al. 2022](#orgbfef01e)) for more details.

We provide an implementation of ColBERT and PLAID, exposed using the classes `PLAIDDocumentStore` and
`ColBERTRetriever`, together with a trained model, see [ColBERT-NQ](https://huggingface.co/Intel/ColBERT-NQ). The
document store class requires the following arguments:

- `collection_path` is the path to the documents collection, in the form of a TSV file with columns being
"id,content,title" where the title is optional.
- `checkpoint_path` is the path for the encoder model, needed to encode queries into vectors at run time. Could be a
local path to a model or a model hosted on HuggingFace hub. In order to use our trained **model** based on
NaturalQuestions, provide the path `Intel/ColBERT-NQ`; see [Model
Hub](https://huggingface.co/Intel/ColBERT-NQ) for more details.
- `index_path` location of the indexed documents. The index contains the optimized and compressed vector representation
of all the documents. Index can be created by the user given a collection and a checkpoint, or can be specified via a
path.

**Updated:** new feature that enables adding and removing documents from a given index. Example usage:

```python
index_updater = IndexUpdater(config, searcher, checkpoint)

added_pids = index_updater.add(passages)  # Adding passages
index_updater.remove(pids)                # Removing passages
searcher.search()                         # Search now reflects the added & removed passages

index_updater.persist_to_disk()           # Persist changes to disk
```

#### PLAID Requirements

If GPU is to be used, it should be of type RTX 3090 or newer (Ampere) and PyTorch should be installed with CUDA support, e.g.:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## fastRAG running LLMs with Habana Gaudi (DL1) and Gaudi 2

fastRAG includes Intel Habana Gaudi support for running LLM as generators in pipelines.

### Installation

To enable Gaudi support, please follow the installation instructions as specified in [Optimum Habana
](https://github.com/huggingface/optimum-habana.git) guide.

### Usage

We enabled support for running LLMs on Habana Gaudi (DL1) and Habana Gaudi 2 by simply configuring the invocation layer of the PromptModel instance.

See below an example for loading a `PromptModel` with Habana backend:

```python
from fastrag.prompters.invocation_layers.gaudi_hugging_face_inference import GaudiHFLocalInvocationLayer

PrompterModel = PromptModel(
    model_name_or_path= "meta-llama/Llama-2-7b-chat-hf",
    invocation_layer_class=GaudiHFLocalInvocationLayer,
    model_kwargs= dict(
        max_new_tokens=50,
        torch_dtype=torch.bfloat16,
        do_sample=False,
        constant_sequence_length=384
    )
)
```

We provide a detailed [Gaudi Inference](examples/inference_with_gaudi.ipynb) notebook, showing how you can build a RAG pipeline using Gaudi; feel free to try it out!

## fastRAG running LLMs with ONNX-runtime

To run LLM efficiently and quickly on CPUs, we provide a method for running quantized LLMs using the [optimum-intel](https://github.com/huggingface/optimum-intel).
We recommend checking out our [full notebook](examples/rag_with_quantized_llm.ipynb) with all the details, including the quantization and pipeline construction.

### Installation

Run the following command to install our dependencies:

```
pip install -e .[intel]
```

For more information regarding the installation process, we recommend checking out the [optimum-intel](https://github.com/huggingface/optimum-intel) repository.

### LLM Quantization

To quantize a model, we first export it the model to the ONNX format, and then use a quantizer to save the quantized version of our model:

```python
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import os

model_name = 'my_llm_model_name'
converted_model_path = "my/local/path"

model = ORTModelForCausalLM.from_pretrained(model_name, export=True)
model.save_pretrained(converted_model_path)

model = ORTModelForCausalLM.from_pretrained(converted_model_path, session_options=session_options)
qconfig = AutoQuantizationConfig.avx2(is_static=False)
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(save_dir=os.path.join(converted_model_path, 'quantized'), quantization_config=qconfig)
```

### Loading the Quantized Model

Now that our model is quantized, we can load it in our framework, by specifying the ```ORTInvocationLayer``` invocation layer.

```python
PrompterModel = PromptModel(
    model_name_or_path= "my/local/path/quantized",
    invocation_layer_class=ORTInvocationLayer,
)
```

## fastRAG Running RAG Pipelines with LLMs on a Llama CPP backend

To run LLM effectively on CPUs, especially on client side machines, we offer a method for running LLMs using the [llama-cpp](https://github.com/ggerganov/llama.cpp).
We recommend checking out our [tutorial notebook](examples/client_inference_with_Llama_cpp.ipynb) with all the details, including processes such as downloading GGUF models.

### Installation

Run the following command to install our dependencies:

```
pip install -e .[llama_cpp]
```

For more information regarding the installation process, we recommend checking out the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) repository.


### Loading the Model

Now that our model is downloaded, we can load it in our framework, by specifying the ```LlamaCPPInvocationLayer``` invocation layer.

```python
PrompterModel = PromptModel(
    model_name_or_path= "models/marcoroni-7b-v3.Q4_K_M.gguf",
    invocation_layer_class=LlamaCPPInvocationLayer,
    model_kwargs= dict(
        max_new_tokens=100
    )
)
```


## Optimized Embedding Models

Bi-encoder Embedders are key components of Retrieval Augmented Generation pipelines. Mainly used for indexing documents and for online re-ranking. We provide support for quantized `int8` models that have low latency and high throughput, using [`optimum-intel`](https://github.com/huggingface/optimum-intel) framework.

For a comprehensive overview, instructions for optimizing existing models and usage information we provide a dedicated [readme.md](scripts/optimizations/embedders/README.md).

We integrated the optimized embedders into the following two components:

- [`QuantizedBiEncoderRanker`](fastrag/rankers/quantized_bi_encoder.py) - bi-encoder rankers; encodes the documents provided in the input and re-orders according to query similarity.
- [`QuantizedBiEncoderRetriever`](fastrag/retrievers/optimized.py) - a bi-encoder retriever; encodes documents into vectors given a vectors store engine.

**NOTE**: For optimal performance we suggest following the important notes in the dedicated [readme.md](scripts/optimizations/embedders/README.md).

## Fusion-In-Decoder

<image align="right" src="assets/fid.png" width="500">

The Fusion-In-Decoder model (FiD in short) is a transformer-based generative model, that is based on the T5 architecture. For our setting, the model answers a question given a question and relevant information about it. Thus, given a query and a collection of documents, it encodes the question combined with each of the documents simultaneously, and later uses all encoded documents at once to generate each token of the answer at a time. See ([Izacard and Grave 2021](#org330b3f6)) for more details.

We provide an implementation of FiD as an invocation layer ([`FiDHFLocalInvocationLayer`](fastrag/prompters/invocation_layers/fid.py)) for a LLM and an [example notebook](examples/fid_promping.ipynb) of a RAG pipeline.

To fine-tune your own an FiD model, you can use our training script here: [Training FiD](scripts/training/train_fid.py)

The following is an example command, with the standard parameters for training the FiD model:

```
python scripts/training/train_fid.py
--do_train \
--do_eval \
--output_dir output_dir \
--train_file path/to/train_file \
--validation_file path/to/validation_file \
--passage_count 100 \
--model_name_or_path t5-base \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--seed 42 \
--gradient_accumulation_steps 8 \
--learning_rate 0.00005 \
--optim adamw_hf \
--lr_scheduler_type linear \
--weight_decay 0.01 \
--max_steps 15000 \
--warmup_step 1000 \
--max_seq_length 250 \
--max_answer_length 20 \
--evaluation_strategy steps \
--eval_steps 2500 \
--eval_accumulation_steps 1 \
--gradient_checkpointing \
--bf16 \
--bf16_full_eval
```

## References

<a id="shiREPLUGRetrievalAugmentedBlackBox2023"></a>Shi, Weijia, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. “REPLUG: Retrieval-Augmented Black-Box Language Models.” arXiv. <https://doi.org/10.48550/arXiv.2301.12652>.

<a id="orgbfef01e"></a>Santhanam, Keshav, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. “PLAID: An Efficient Engine for Late Interaction Retrieval.” arXiv. <https://doi.org/10.48550/arXiv.2205.09707>.

<a id="org330b3f5"></a>Santhanam, Keshav, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. “ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.” arXiv. <https://doi.org/10.48550/arXiv.2112.01488>.

<a id="org330b3f6"></a>Izacard, Gautier,  and Edouard Grave. 2021. “Leveraging Passage  Retrieval with Generative Models for Open Domain Question Answering.” arXiv. <https://doi.org/10.48550/arXiv.2007.01282>.
