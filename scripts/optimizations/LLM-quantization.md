# Running Quantized LLMs on ONNX Runtime

To run LLM efficiently and quickly on CPUs, we provide a method for running quantized LLMs using the [optimum-intel](https://github.com/huggingface/optimum-intel).
We recommend checking out our [full notebook](../../examples/rag_with_quantized_llm.ipynb) with all the details, including the quantization and pipeline construction.

## Installation

Run the following command to install our dependencies:

```
pip install -e .[intel]
```

For more information regarding the installation process, we recommend checking out the [optimum-intel](https://github.com/huggingface/optimum-intel) repository.

## LLM Quantization

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

## Loading the Quantized Model

Now that our model is quantized, we can load it in our framework, by using the ```ORTGenerator``` generator.

```python
generator = ORTGenerator(
    model="my/local/path/quantized",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 100,
    }
)
```

We also included some additional parameters for the [ort.SessionOptions](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session_options.html) for loading the model:

* graph_optimization_level: **str**
* intra_op_num_threads: **int**
* session_config_entries: **dict**

For example:

```python
generator = ORTGenerator(
    model="my/local/path/quantized",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 100,
    }
    huggingface_pipeline_kwargs=dict(
        graph_optimization_level="ORT_ENABLE_ALL",
        intra_op_num_threads=6,
        session_config_entries={
            "session.intra_op_thread_affinities": "3,4;5,6;7,8;9,10;11,12"
        }
    )
)
```

We also include an example YAML file of deploying the ORT model as an API [here](../../config/doc_chat_ort.yaml).

For more information regarding the different settings, we refer to the [ort.SessionOptions](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session_options.html) documentation.
