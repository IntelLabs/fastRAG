# Optimizing Bi-Encoder Embedders with [`optimum-intel`](https://github.com/huggingface/optimum-intel)

Embedders are key components of Retrieval Augmented Generation pipelines. Mainly used for indexing documents and for online re-ranking.

We showcase a recipe to improve the performance (latency (batch size=1), throughput) of embedders using `optimum-intel` for quantizing a model to int8 and running the optimized model using IPEX backend.

The produced quantized models can be used with fastRAG `QuantizedBiEncoderRanker` and `QuantizedBiEncoderRetriever`

## How to quantize?

Steps required for quantization:

- Installing required python packages (mainly `optimum-intel` and `intel-extension-for-transformers`).

    ```bash
    pip install -r requirements.txt
    ```

- Quantizing a model by running `python quantize_embedder.py --quantize`:
  - with provided calibration data and a model from Hugging Face model hub
- Benchmarking a quantized model or a vanilla (non-quantized) model using `quantize_embedder.py --benchmark`:
  - running an evaluation on a subset of Reranking or Retrieval tasks of [MTEB](https://github.com/embeddings-benchmark/mteb) benchmark suite.

### Examples

Quantize [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) using 100 samples:

```bash
python quantize_embedder.py --quantize --model_name BAAI/bge-small-en-v1.5 --output_path quantized_model/ --sample_size 100
```

Benchmark a quantized model on the Reranking tasks of MTEB. Use `--benchmark` to run only benchmarking and `--opt` for benchmarking a quantized model:

```bash
python quantize_embedder.py --benchmark --opt --model_name quantized_model/ --task rerank
```

## Running a quantized model

Running inference using a quantized model is similar to Hugging Face API. We use `optimum-intel` Auto models for loading a model.

**Loading a model:**

```python
from optimum.intel import IPEXModel

model = IPEXModel.from_pretrained("Intel/bge-small-en-v1.5-rag-int8-static")
```

**Inference with auto-mixed precision (bf16):**

```python
with torch.no_grad():
    outputs = model(**inputs)
```

## Benchmarking speed and latency

Quantized and vanilla Hugging Face models can be benchmarks for latency (batch size = 1) and throughput using the provided `benchmark_speed.py` script.

The benchmarking script uses [Aim](https://github.com/aimhubio/aim) to log experiment settings and metrics. To install `Aim` run the following command `pip install aim`; and `aim up` to launch the UI.

The script can benchmark the following model backends:

- Vanilla PyTorch
- IPEX w/ and w/o bf16
- IPEX torch-script (traced model) w/ and w/o bf16
- Optimum-intel quantized model with IPEX backend

### Running instructions

The benchmarking script has several argument that define the benchmark:

- `--model-name`: path to a quantized model or a Huggingface model name
- `--mode`: `inc`, `hf`, `ipex`, `ipex-ts`; model types
- `--bf16`: activate `bf16` inference
- `--samples`: the number of samples to run in the benchmark
- `--bs`: batch size
- `--seq-len`: the sequence length of each sample when running the benchmarks
- `--warmup`: the number of warmup cycles to do before measuring latency/throughput

To effectively utilize the CPU resources when running on a Intel Xeon processors, we should limit the processes to run on a single socket. This can be done by using `numactl`.
In addition, it is recommended to use TCMalloc for better performance when accessing commonly-used objects.

#### Using `numactl`

How to install:

```bash
sudo apt-get update
sudo apt-get install numactl
```

How to run:

- `-C` : specify core indexes to use; for example (0-31) instructs to use cores 0 to 31 (32 in total)

```bash
numactl -C 0-31 -m 0 python script.py args ...
```

#### TCMalloc

Further info on TCMalloc is available [here](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html#tcmalloc):

- How to install [TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md#build-tcmalloc).
- Once installed, expose via an `env`: `export LD_PRELOAD=/path/to/libtcmalloc.so`
