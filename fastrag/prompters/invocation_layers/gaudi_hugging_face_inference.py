import copy
import logging
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.handlers import (
    DefaultTokenStreamingHandler,
    HFTokenStreamingHandler,
)
from haystack.nodes.prompt.invocation_layer.hugging_face import (
    HFLocalInvocationLayer,
    StopWordsCriteria,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.pipelines import get_task

logger = logging.getLogger(__name__)

with LazyImport(
    "habana_frameworks or optimum.habana are not installed."
    "Install pytorch support using this guide: https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pytorch-installation "
    "Install Optimum-habana by running 'pip install --upgrade-strategy eager optimum[habana]'"
) as habana_import:
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()


class GaudiHFLocalInvocationLayer(HFLocalInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained model from Hugging Face,
    and loads it into an HPU device, including ad-hoc optimizations.
    """

    def __init__(
        self,
        model_name_or_path: str = "Intel/neural-chat-7b-v2",
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        habana_import.check()
        PromptModelInvocationLayer.__init__(self, model_name_or_path)
        self.use_auth_token = use_auth_token

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to Hugging Face pipeline documentation
        # Do not use `device_map` AND `device` at the same time as they will conflict
        model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "model_kwargs",
                "trust_remote_code",
                "revision",
                "feature_extractor",
                "tokenizer",
                "config",
                "use_fast",
                "torch_dtype",
                "device_map",
                "generation_kwargs",
                "model_max_length",
                "stream",
                "stream_handler",
            ]
            if key in kwargs
        }
        self.constant_sequence_length = None
        if "constant_sequence_length" in kwargs:
            self.constant_sequence_length = kwargs.pop("constant_sequence_length")

        # flatten model_kwargs one level
        if "model_kwargs" in model_input_kwargs:
            mkwargs = model_input_kwargs.pop("model_kwargs")
            model_input_kwargs.update(mkwargs)

        # save stream settings and stream_handler for pipeline invocation
        self.stream_handler = model_input_kwargs.pop("stream_handler", None)
        self.stream = model_input_kwargs.pop("stream", False)

        # save generation_kwargs for pipeline invocation
        self.generation_kwargs = model_input_kwargs.pop("generation_kwargs", {})
        model_max_length = model_input_kwargs.pop("model_max_length", None)

        torch_dtype = model_input_kwargs.get("torch_dtype")
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if "torch." in torch_dtype:
                    torch_dtype_resolved = getattr(torch, torch_dtype.strip("torch."))
                elif torch_dtype == "auto":
                    torch_dtype_resolved = torch_dtype
                else:
                    raise ValueError(
                        f"torch_dtype should be a torch.dtype, a string with 'torch.' prefix or the string 'auto', got {torch_dtype}"
                    )
            elif isinstance(torch_dtype, torch.dtype):
                torch_dtype_resolved = torch_dtype
            else:
                raise ValueError(f"Invalid torch_dtype value {torch_dtype}")
            model_input_kwargs["torch_dtype"] = torch_dtype_resolved

        # If task_name is not provided, get the task name from the model name or path (uses HFApi)
        if "task_name" in kwargs:
            self.task_name = kwargs.get("task_name")
        else:
            self.task_name = get_task(model_name_or_path, use_auth_token=use_auth_token)

        self.device = "hpu"

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        model = model.eval().to(self.device)
        model = wrap_in_hpu_graph(model)
        model.generation_config.use_cache = True

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        if self.constant_sequence_length:
            tokenizer.padding_side = (
                "left"  # since padding will probably be added, we insert it on the left
            )

        # to preserve the original syntax of the invocation layer, with a pipe object:
        self.pipe = Namespace(model=model, tokenizer=tokenizer, device=model.device)

        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise HFLocalInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

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

    def tokenize(self, input_sentences):
        # Tokenization

        if self.constant_sequence_length:
            tokenization_args = {
                "padding": "max_length",
                "max_length": self.constant_sequence_length,
            }
        else:
            tokenization_args = {"padding": True}

        input_tokens = self.pipe.tokenizer.batch_encode_plus(
            input_sentences, return_tensors="pt", **tokenization_args
        ).to(self.device)

        return input_tokens

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated texts using the local Hugging Face transformers model
        :return: A list of generated texts.

        Note: Only kwargs relevant to Text2TextGenerationPipeline and TextGenerationPipeline are passed to
        Hugging Face as model_input_kwargs. Other kwargs are ignored.
        """
        output: List[Dict[str, str]] = []
        stop_words = kwargs.pop("stop_words", None)
        top_k = kwargs.pop("top_k", None)
        # either stream is True (will use default handler) or stream_handler is provided for custom handler
        stream = (
            kwargs.get("stream", self.stream)
            or kwargs.get("stream_handler", self.stream_handler) is not None
        )
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            # Consider only Text2TextGenerationPipeline and TextGenerationPipeline relevant, ignore others
            # For more details refer to Hugging Face Text2TextGenerationPipeline and TextGenerationPipeline
            # documentation

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

            is_text_generation = "text-generation" == self.task_name
            # Prefer return_full_text is False for text-generation (unless explicitly set)
            # Thus only generated text is returned (excluding prompt)
            if is_text_generation and "return_full_text" not in model_input_kwargs:
                model_input_kwargs["return_full_text"] = False
                model_input_kwargs["max_new_tokens"] = self.max_length
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
            # max_new_tokens is used for text-generation and max_length for text2text-generation
            if is_text_generation:
                model_input_kwargs["max_new_tokens"] = model_input_kwargs.pop(
                    "max_length", self.max_length
                )
            else:
                model_input_kwargs["max_length"] = self.max_length

            if stream:
                stream_handler: TokenStreamingHandler = kwargs.pop(
                    "stream_handler", DefaultTokenStreamingHandler()
                )
                model_input_kwargs["streamer"] = HFTokenStreamingHandler(
                    self.pipe.tokenizer, stream_handler
                )

            input_tokens = self.tokenize([prompt])

            gen_conf = copy.deepcopy(self.pipe.model.generation_config)
            for gen_k, gen_v in model_input_kwargs.items():
                if gen_k in ["min_length"]:
                    continue
                setattr(gen_conf, gen_k, gen_v)

            if "max_new_tokens" in generation_kwargs:
                setattr(gen_conf, "max_new_tokens", generation_kwargs["max_new_tokens"])

            out_tokens = self.pipe.model.generate(
                **input_tokens,
                generation_config=gen_conf,
                lazy_mode=True,
                hpu_graphs=True,
            ).cpu()

            out_text = self.pipe.tokenizer.batch_decode(
                out_tokens[:, input_tokens["input_ids"].shape[1] :]
            )

            # If EOS token has been produced, ensure only the output previous to it is returned.
            output = [
                {"generated_text": text.split(self.pipe.tokenizer.eos_token)[0]}
                for text in out_text
            ]

        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

        return generated_texts
