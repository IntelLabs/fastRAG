from typing import Any, Dict, List, Literal, Optional

from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret
from transformers import AutoConfig, AutoTokenizer

with LazyImport("Run 'pip install -e .[optimum]'") as onnx_import:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM


class ORTGenerator(HuggingFaceLocalGenerator):
    """
    Generator based on a Hugging Face model loaded with ORT.

    This component provides an interface to generate text using a compressed Hugging Face model that runs locally using ORT.

    Usage example:
    ```python

    generator = ORTGenerator(
        model="microsoft/phi-2",
        task="text-generation",
        generation_kwargs={
            "max_new_tokens": 100,
            "temperature": 0.1,
        }
    )

    print(generator.run("Who is the best American actor?"))
    # {'replies': ['John Cusack']}
    ```
    """

    def __init__(
        self,
        model: str = "microsoft/phi-2",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
    ):
        """
        Creates an instance of a OpenVINOGenerator.

        :param model: The name or path of a Hugging Face model for text generation,
        :param task: The task for the Hugging Face pipeline.
            Possible values are "text-generation" and "text2text-generation".
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If the task is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
            If not specified, the component will attempt to infer the task from the model name,
            calling the Hugging Face Hub API.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If the token is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information:
            - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
            - [transformers.GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            See Hugging Face's [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task)
            for more information on the available kwargs.
            In this dictionary, you can also include `model_kwargs` to specify the kwargs for model initialization:
            [transformers.PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
        :param stop_words: A list of stop words. If any one of the stop words is generated, the generation is stopped.
            If you provide this parameter, you should not specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, it's important to make sure your prompt has no stop words.
        """
        onnx_import.check()
        super().__init__(
            model=model,
            task=task,
            device=device,
            token=token,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs,
            stop_words=stop_words,
        )
        self.ort_classes = [
            ORTModelForCausalLM,
            ORTModelForSeq2SeqLM,
        ]

        model_instance, tokenizer = self.get_ort_model(model, **self.huggingface_pipeline_kwargs)

        self.huggingface_pipeline_kwargs["model"] = model_instance
        self.huggingface_pipeline_kwargs["tokenizer"] = tokenizer

    def get_ort_model(self, model_name_or_path, **kwargs):
        """We also included some additional parameters for the [ort.SessionOptions](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session_options.html) for loading the model:

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
        """
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        ort_model_class = self.get_ort_class(model_config)

        graph_optimization_level = kwargs.pop("graph_optimization_level", "ORT_ENABLE_ALL")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, graph_optimization_level
        )
        session_options.intra_op_num_threads = kwargs.pop("intra_op_num_threads", 6)

        session_config_entries = kwargs.pop("session_config_entries", {})

        for session_config_entry_key, session_config_entry_value in session_config_entries.items():
            session_options.add_session_config_entry(
                session_config_entry_key, session_config_entry_value
            )

        self.huggingface_pipeline_kwargs = kwargs

        model = ort_model_class.from_pretrained(
            model_name_or_path, session_options=session_options, **kwargs
        )
        return model, AutoTokenizer.from_pretrained(model_name_or_path)

    def get_ort_class(self, config):
        for current_class in self.ort_classes:
            class_suffix = current_class.__name__.replace("ORTModel", "")

            # If the architecture matches the ORT class, return it
            if config.architectures[0].endswith(class_suffix):
                # return the class and the task
                return current_class

        # No match has been found
        raise Exception("No ORT class found for the model.")
