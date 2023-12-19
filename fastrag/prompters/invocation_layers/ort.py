import logging
from typing import Optional, Union

from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.utils import get_task
from transformers import AutoConfig, AutoTokenizer, Pipeline, pipeline

logger = logging.getLogger(__name__)

from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer.hugging_face import HFLocalInvocationLayer

with LazyImport("Run 'pip install -e .[optimum]'") as onnx_import:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM


class ORTInvocationLayer(HFLocalInvocationLayer):
    """
    A subclass of the HFLocalInvocationLayer class. It loads an onnx-quantized pre-trained model.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        """
        Creates an instance of ORTInvocationLayer used to invoke local ORT quantized models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text can have.
        :param use_auth_token: The token to use as HTTP bearer authorization for remote files.
        :param use_gpu: Whether to use GPU for inference.
        :param device: The device to use for inference.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of HFLocalInvocationLayer might receive some unrelated
        kwargs. Only kwargs relevant to the HFLocalInvocationLayer are considered. The list of supported kwargs
        includes: "task", "model", "config", "tokenizer", "feature_extractor", "revision", "use_auth_token",
        "device_map", "device", "torch_dtype", "trust_remote_code", "model_kwargs", and "pipeline_class".
        For more details about pipeline kwargs in general, see
        Hugging Face [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline).

        This layer supports two additional kwargs: generation_kwargs and model_max_length.

        The generation_kwargs are used to customize text generation for the underlying pipeline. See Hugging
        Face [docs](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
        for more details.

        The model_max_length is used to specify the custom sequence length for the underlying pipeline.
        """
        onnx_import.check()
        PromptModelInvocationLayer.__init__(self, model_name_or_path)

        self.map_classes_to_task = {
            ORTModelForCausalLM: "text-generation",
            ORTModelForSeq2SeqLM: "text2text-generation",
        }

        self.use_auth_token = use_auth_token

        # # save stream settings and stream_handler for pipeline invocation
        self.stream_handler = kwargs.get("stream_handler", None)
        self.stream = kwargs.get("stream", False)

        # save generation_kwargs for pipeline invocation
        self.generation_kwargs = kwargs.get("generation_kwargs", {})

        # If task_name is not provided, get the task name from the model name or path (uses HFApi)
        self.task_name = (
            kwargs.get("task_name")
            if "task_name" in kwargs
            else get_task(model_name_or_path, use_auth_token=use_auth_token)
        )
        # we check in supports class method if task_name is supported but here we check again as
        # we could have gotten the task_name from kwargs
        if self.task_name not in ["text2text-generation", "text-generation"]:
            raise ValueError(
                f"Task name {self.task_name} is not supported. "
                f"We only support text2text-generation and text-generation tasks."
            )
        pipeline_kwargs = self._prepare_pipeline_kwargs(
            task=self.task_name,
            model_name_or_path=model_name_or_path,
            use_auth_token=use_auth_token,
            **kwargs,
        )

        ort_model_args = self.get_ort_model(model_name_or_path, **kwargs)
        pipeline_kwargs.update(ort_model_args)

        # create the transformer pipeline
        self.pipe: Pipeline = pipeline(**pipeline_kwargs)

        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise HFLocalInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

        model_max_length = kwargs.get("model_max_length", None)
        # we allow users to override the tokenizer's model_max_length because models like T5 have relative positional
        # embeddings and can accept sequences of more than 512 tokens
        if model_max_length is not None:
            self.pipe.tokenizer.model_max_length = model_max_length

        if self.max_length > self.pipe.tokenizer.model_max_length:
            logger.warning(
                "The max_length %s is greater than model_max_length %s. This might result in truncation of the "
                "generated text. Please lower the max_length (number of answer tokens) parameter!",
                self.max_length,
                self.pipe.tokenizer.model_max_length,
            )

    def get_ort_model(self, model_name_or_path, **kwargs):
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        ort_model_class, task = self.get_ort_class(model_config)

        graph_optimization_level = kwargs.get("graph_optimization_level", "ORT_ENABLE_ALL")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, graph_optimization_level
        )
        session_options.intra_op_num_threads = kwargs.get("intra_op_num_threads", 6)

        session_config_entries = kwargs.get("session_config_entries", {})

        for session_config_entry_key, session_config_entry_value in session_config_entries.items():
            session_options.add_session_config_entry(
                session_config_entry_key, session_config_entry_value
            )

        model = ort_model_class.from_pretrained(model_name_or_path, session_options=session_options)
        return {
            "model": model,
            "task": task,
            "tokenizer": AutoTokenizer.from_pretrained(model_name_or_path),
        }

    def get_ort_class(self, config):
        for current_class, current_task in self.map_classes_to_task.items():
            class_suffix = current_class.__name__.replace("ORTModel", "")

            # If the architecture matches the ORT class, return it
            if config.architectures[0].endswith(class_suffix):
                # return the class and the task
                return current_class, current_task

        # No match has been found
        raise Exception("No ORT class found for the model.")
