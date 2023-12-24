# Running RAG Pipelines with LLMs on a Llama CPP backend

To run LLM effectively on CPUs, especially on client side machines, we offer a method for running LLMs using the [llama-cpp](https://github.com/ggerganov/llama.cpp).
We recommend checking out our [tutorial notebook](../../examples/client_inference_with_Llama_cpp.ipynb) with all the details, including processes such as downloading GGUF models.

## Installation

Run the following command to install our dependencies:

```
pip install -e .[llama_cpp]
```

For more information regarding the installation process, we recommend checking out the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) repository.

## Downloading GGUF models

In order to use LlamaCPP, download a gguf model, optimal for llama cpp inference:

```
huggingface-cli download TheBloke/Marcoroni-7B-v3-GGUF marcoroni-7b-v3.Q4_K_M.gguf --local-dir ./models --local-dir-use-symlinks False
```

## Loading the Model

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
