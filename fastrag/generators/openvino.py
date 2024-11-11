from typing import Any, Callable, Dict, List, Literal, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import (
    ComponentDevice,
    Secret,
    deserialize_callable,
    deserialize_secrets_inplace,
    serialize_callable,
)
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs
from transformers import AutoConfig, AutoTokenizer

with LazyImport("Install openvino using 'pip install -e .[openvino]'") as ov_import:
    from optimum.intel.openvino import OVModelForCausalLM

DEFAULT_OV_CONFIG = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}


@component
class OpenVINOGenerator(HuggingFaceLocalGenerator):
    """
    Generator based on a Hugging Face model loaded with OpenVINO.

    This component provides an interface to generate text using a compressed Hugging Face model that runs locally using OpenVINO.

    Usage example:
    ```python

    generator = OpenVINOGenerator(
        model="microsoft/phi-2",
        compressed_model_dir=openvino_compressed_model_path,
        device_openvino="CPU",
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
        compressed_model_dir: str = None,
        device_openvino: str = "CPU",
        ov_config: dict = DEFAULT_OV_CONFIG,
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Creates an instance of a OpenVINOGenerator.

        :param model: The name or path of a Hugging Face model for text generation,
        :param compressed_model_dir: The name or path of the compressed model,
        :param task: The task for the Hugging Face pipeline.
            Possible values are "text-generation" and "text2text-generation".
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If the task is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
            If not specified, the component will attempt to infer the task from the model name,
            calling the Hugging Face Hub API.
        :param device_openvino: The Intel device on which the model will be loaded, run 'openvino.Core()' to view the available devices. Defaults to CPU.
        :param ov_config: The OpenVINO configuration dictionary.
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
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        ov_import.check()
        super(OpenVINOGenerator, self).__init__(
            model=model,
            task=task,
            device=device,
            token=token,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        self.model = model
        self.compressed_model_dir = compressed_model_dir
        self.device_openvino = device_openvino
        self.ov_config = ov_config

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.pipeline is None:
            ov_model = OVModelForCausalLM.from_pretrained(
                self.compressed_model_dir,
                device=self.device_openvino,
                ov_config=self.ov_config,
                config=AutoConfig.from_pretrained(
                    self.compressed_model_dir, trust_remote_code=True
                ),
                trust_remote_code=True,
            )

            self.huggingface_pipeline_kwargs["model"] = ov_model
            self.huggingface_pipeline_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(
                self.model
            )
            super(OpenVINOGenerator, self).warm_up()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = (
            serialize_callable(self.streaming_callback) if self.streaming_callback else None
        )
        serialization_dict = default_to_dict(
            self,
            huggingface_pipeline_kwargs=self.huggingface_pipeline_kwargs,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
            stop_words=self.stop_words,
            token=self.token.to_dict() if self.token else None,
            model=self.model,
            compressed_model_dir=self.compressed_model_dir,
            device_openvino=self.device_openvino,
            ov_config=self.ov_config,
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"][
            "huggingface_pipeline_kwargs"
        ]
        huggingface_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return serialization_dict
