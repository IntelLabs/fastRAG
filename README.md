> [!IMPORTANT]
> We're migrating `fastRAG` to Haystack v2.0 API and will release a major update soon. Stay tuned!

<div align="center">
    <img src="assets/fastrag_header.png" width="300"/>

---

<h4 align="center">
    <p>Build and explore efficient retrieval-augmented generative models and applications</p>
</h4>

:round_pushpin: <a href="#round_pushpin-installation">Installation</a> • :rocket: <a href="components.md">Components</a> • :books: <a href="examples.md">Examples</a> • :red_car: <a href="getting_started.md">Getting Started</a> • :pill: <a href="demo/README.md">Demos</a> • :pencil2: <a href="scripts/README.md">Scripts</a> • :bar_chart: <a href="benchmarks/README.md">Benchmarks</a>


</div>

fast**RAG** is a research framework for ***efficient*** and ***optimized*** retrieval augmented generative pipelines,
incorporating state-of-the-art LLMs and Information Retrieval. fastRAG is designed to empower researchers and developers
with a comprehensive tool-set for advancing retrieval augmented generation.

Comments, suggestions, issues and pull-requests are welcomed! :heart:

## :mega: Updates

- **2024-04**: [(Extra Demo)](extras/rag_on_client.ipynb) **Chat with your documents on Intel Meteor Lake iGPU**.
- **2023-12**: Gaudi2, ONNX runtime and LlamaCPP support; Optimized Embedding models; Multi-modality and Chat demos; [REPLUG](https://arxiv.org/abs/2301.12652) text generation.
- **2023-06**: ColBERT index modification: adding/removing documents; see [IndexUpdater](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/index_updater.py).
- **2023-05**: [RAG with LLM and dynamic prompt synthesis example](examples/rag-prompt-hf.ipynb).
- **2023-04**: Qdrant `DocumentStore` support.

## Key Features

- **Optimized RAG**: Build RAG pipelines with SOTA efficient components for greater compute efficiency.
- **Optimized for Intel Hardware**: Leverage [Intel extensions for PyTorch (IPEX)](https://github.com/intel/intel-extension-for-pytorch), [🤗 Optimum Intel](https://github.com/huggingface/optimum-intel) and [🤗 Optimum-Habana](https://github.com/huggingface/optimum-habana) for *running as optimal as possible* on Intel® Xeon® Processors and Intel® Gaudi® AI accelerators.
- **Customizable**: fastRAG is built using [Haystack](https://github.com/deepset-ai/haystack) and HuggingFace. All of fastRAG's components are 100% Haystack compatible.

## :rocket: Components

For a brief overview of the various unique components in fastRAG refer to the [Components Overview](components.md) page.

<div class="tg-wrap" align="center">
<table style="undefined;table-layout: fixed; width: 600px; text-align: center;">
<colgroup>
<!-- <col style="width: 229px"> -->
<!-- <col style="width: 238px"> -->
</colgroup>
<tbody>
  <tr>
    <td colspan="2"><strong><em>LLM Backends</em></td>
  </tr>
  <tr>
    <td><a href="components.md#fastrag-running-llms-with-habana-gaudi-(dl1)-and-gaudi-2">Intel Gaudi Accelerators</a></td>
    <td><em>Running LLMs on Gaudi 2</td>
  </tr>
  <tr>
    <td><a href="components.md#fastrag-running-llms-with-onnx-runtime">ONNX Runtime</a></td>
    <td><em>Running LLMs with optimized ONNX-runtime</td>
  </tr>
  <tr>
    <td><a href="components.md#fastrag-running-rag-pipelines-with-llms-on-a-llama-cpp-backend">Llama-CPP</a></td>
    <td><em>Running RAG Pipelines with LLMs on a Llama CPP backend</td>
  </tr>
  <tr>
    <td colspan="2"><strong><em>Optimized Components</em></td>
  </tr>
  <tr>
    <td><a href="scripts/optimizations/embedders/README.md">Embedders</a></td>
    <td>Optimized int8 bi-encoders</td>
  </tr>
  <tr>
    <td><a href="scripts/optimizations/reranker_quantization/quantization.md">Rankers</a></td>
    <td>Optimized/sparse cross-encoders</td>
  </tr>
  <tr>
    <td colspan="2"><strong><em>RAG-efficient Components</em></td>
  </tr>
  <tr>
    <td><a href="components.md#ColBERT-v2-with-PLAID-Engine">ColBERT</a></td>
    <td>Token-based late interaction</td>
  </tr>
  <tr>
    <td><a href="components.md#Fusion-In-Decoder">Fusion-in-Decoder (FiD)</a></td>
    <td>Generative multi-document encoder-decoder</td>
  </tr>
  <tr>
    <td><a href="components.md#REPLUG">REPLUG</a></td>
    <td>Improved multi-document decoder</td>
  </tr>
  <tr>
    <td><a href="components.md#ColBERT-v2-with-PLAID-Engine">PLAID</a></td>
    <td>Incredibly efficient indexing engine</td>
  </tr>
</tbody>
</table></div>

## :round_pushpin: Installation

Preliminary requirements:

- **Python** 3.8 or higher.
- **PyTorch** 2.0 or higher.

To set up the software, clone the project and run the following, preferably in a newly created virtual environment:

```bash
pip install .
```

There are several dependencies to consider, depending on your specific usage:

```bash
# Additional engines/components
pip install .[intel]               # Intel optimized backend [Optimum-intel, IPEX]
pip install .[elastic]             # Support for ElasticSearch store
pip install .[qdrant]              # Support for Qdrant store
pip install .[colbert]             # Support for ColBERT+PLAID; requires FAISS
pip install .[faiss-cpu]           # CPU-based Faiss library
pip install .[faiss-gpu]           # GPU-based Faiss library
pip install .[knowledge_graph]     # Libraries for working with spacy and KG

# User interface (for demos)
pip install .[ui]

# Benchmarking
pip install .[benchmark]

# Development tools
pip install .[dev]
```

## License

The code is licensed under the [Apache 2.0 License](LICENSE).

## Disclaimer

This is not an official Intel product.
