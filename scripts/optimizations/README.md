# Optimizing Components of fastRAG on Intel Hardware

Models can be further optimized through software frameworks to improve latency and throughput. Software packages such as [`optimum-intel`](https://github.com/huggingface/optimum-intel) developed by Intel and partners are designed to leverage the CPU extensions found in the most recent Intel processors.
Transformer-based models can undergo quantization, sparsification, or enhancement through knowledge distillation by utilizing the [`optimum-intel`](https://github.com/huggingface/optimum-intel) library.

## Quantization

Quantization is a process that minimizes both computational overhead and memory footprint during inference. This is achieved by adopting lower-precision data types, such as 8-bit integers (int8), instead of the standard 32-bit floating-point numbers (float32) to represent model weights and activations. To facilitate these optimizations, frameworks like the [Intel Extension for Pytorch](https://github.com/intel/intel-extension-for-pytorch) and [`optimum-intel`](https://github.com/huggingface/optimum-intel) provide specialized support for the latest Intel CPU features.

**Why should we optimize using quantization?**

Reduction in bit count leads to a model that requires less memory storage, potentially reduces energy consumption, and enables faster operations, such as matrix multiplication, through integer arithmetic.

## Available Optimizations

|                                                                     | framework                   | backend |
|---------------------------------------------------------------------|-----------------------------|---------|
| [LLM Quantization](LLM-quantization.md)                             | `optimum-intel`             | CPU     |
| [Bi-encoder Quantization](embedders/README.md)                      | `optimum-intel`             | CPU     |
| [Cross-encoder Quantization](reranker_quantization/quantization.md) | `neural-compressor`, `ipex` | CPU     |
