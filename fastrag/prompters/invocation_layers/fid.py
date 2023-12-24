import logging
from typing import Any, Dict, List, Optional, Union

import torch
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.handlers import (
    DefaultTokenStreamingHandler,
    HFTokenStreamingHandler,
)
from haystack.nodes.prompt.invocation_layer.hugging_face import (
    HFLocalInvocationLayer,
    StopWordsCriteria,
    initialize_device_settings,
    torch_and_transformers_import,
)
from haystack.nodes.prompt.invocation_layer.utils import get_task
from haystack.nodes.prompt.prompt_template import PromptTemplate
from transformers import AutoTokenizer, GenerationConfig, Pipeline, StoppingCriteriaList, pipeline

from fastrag.readers.FiD import FiDConverter, FusionInDecoderForConditionalGeneration


def remove_template_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return kwargs


logger = logging.getLogger(__name__)


class FiDHFLocalInvocationLayer(HFLocalInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained model from Hugging Face, and
    utilizes the model according to the FiD algorithm (https://arxiv.org/pdf/2007.01282.pdf)
    """

    def __init__(
        self,
        model_name_or_path: str = "Intel/fid_flan_t5_base_nq",
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        **kwargs,
    ):
        torch_and_transformers_import.check()

        PromptModelInvocationLayer.__init__(self, model_name_or_path)

        # We disable the removal of the query and documents from the kwargs,
        # thus allowing the FiD model to process them directly.
        PromptTemplate.remove_template_params = remove_template_params

        self.use_auth_token = use_auth_token

        self.devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )
        if "device" not in kwargs:
            kwargs["device"] = self.devices[0]

        # save stream settings and stream_handler for pipeline invocation
        self.stream_handler = kwargs.get("stream_handler", None)
        self.stream = kwargs.get("stream", False)

        # save generation_kwargs for pipeline invocation
        self.generation_kwargs = kwargs.get("generation_kwargs", {})

        self.task_name = "text2text-generation"

        loading_model_kwargs = kwargs.get("model_kwargs", {})

        kwargs["model"] = FusionInDecoderForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, **loading_model_kwargs
        )

        pipeline_kwargs = self._prepare_pipeline_kwargs(
            task=self.task_name,
            model_name_or_path=model_name_or_path,
            use_auth_token=use_auth_token,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        pipeline_kwargs["tokenizer"] = tokenizer
        del pipeline_kwargs["device"]
        # create the transformer pipeline
        self.pipe: Pipeline = pipeline(**pipeline_kwargs)

        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise HFLocalInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

        model_max_length = kwargs.get("model_max_length", None)
        self.input_converter = FiDConverter(tokenizer.model_max_length)
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

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated texts using the local Hugging Face transformers model
        :return: A list of generated texts.

        Note: Only kwargs relevant to Text2TextGenerationPipeline are passed to
        Hugging Face as model_input_kwargs. Other kwargs are ignored.
        """
        output: List[Dict[str, str]] = []
        stop_words = kwargs.pop("stop_words", None)
        top_k = kwargs.pop("top_k", None)
        # either stream is True (will use default handler) or stream_handler is provided for custom handler
        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        stream = stream or stream_handler is not None

        if "prompt" in kwargs:
            # Consider only Text2TextGenerationPipeline relevant, ignore others
            # For more details refer to Hugging Face Text2TextGenerationPipeline documentation
            model_input_kwargs = {
                key: kwargs[key]
                for key in [
                    "return_tensors",
                    "return_text",
                    "return_full_text",
                    "clean_up_tokenization_spaces",
                    "truncation",
                    "generation_kwargs",
                    "max_new_tokens",
                    "num_beams",
                    "do_sample",
                    "num_return_sequences",
                    "max_length",
                ]
                if key in kwargs
            }

            generation_kwargs = model_input_kwargs.pop("generation_kwargs", self.generation_kwargs)

            if isinstance(generation_kwargs, dict):
                model_input_kwargs.update(generation_kwargs)
            elif isinstance(generation_kwargs, GenerationConfig):
                gen_dict = generation_kwargs.to_diff_dict()
                gen_dict.pop("transformers_version", None)
                model_input_kwargs.update(gen_dict)

            # Prefer return_full_text is False for text-generation (unless explicitly set)
            # Thus only generated text is returned (excluding prompt)

            if stop_words:
                sw = StopWordsCriteria(
                    tokenizer=self.pipe.tokenizer, stop_words=stop_words, device=self.pipe.device
                )
                model_input_kwargs["stopping_criteria"] = StoppingCriteriaList([sw])
            if top_k:
                model_input_kwargs["num_return_sequences"] = top_k
                if "num_beams" not in model_input_kwargs or model_input_kwargs["num_beams"] < top_k:
                    if "num_beams" in model_input_kwargs:
                        logger.warning(
                            "num_beams should not be less than top_k, hence setting it to %s", top_k
                        )
                    model_input_kwargs["num_beams"] = top_k

            model_input_kwargs["max_length"] = model_input_kwargs.pop("max_length", self.max_length)

            if stream:
                stream_handler: TokenStreamingHandler = (
                    stream_handler or DefaultTokenStreamingHandler()
                )
                model_input_kwargs["streamer"] = HFTokenStreamingHandler(
                    self.pipe.tokenizer, stream_handler
                )

            inputs = self.input_converter(
                self.pipe.tokenizer, kwargs["query"], kwargs["documents"]
            ).to(self.pipe.model.device)

            model_input_kwargs = {
                k: v for k, v in model_input_kwargs.items() if k not in ["return_full_text"]
            }

            gen_outputs = self.pipe.model.generate(**inputs, **model_input_kwargs)

            generated_tokens = gen_outputs[0]
            generated_text_only = self.pipe.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            output = [{"generated_text": generated_text_only}]

        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

        if stop_words:
            # Although HF generates text until stop words are encountered unfortunately it includes the stop word
            # We want to exclude it to be consistent with other invocation layers
            for idx, _ in enumerate(generated_texts):
                for stop_word in stop_words:
                    generated_texts[idx] = generated_texts[idx].replace(stop_word, "").rstrip()
        return generated_texts
