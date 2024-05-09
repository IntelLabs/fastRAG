import logging
from typing import Dict, List, Optional, Union

import torch
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.handlers import (
    DefaultTokenStreamingHandler,
    HFTokenStreamingHandler,
)
from haystack.nodes.prompt.invocation_layer.hugging_face import (
    HFLocalInvocationLayer,
    initialize_device_settings,
    torch_and_transformers_import,
)
from haystack.nodes.prompt.invocation_layer.utils import get_task
from transformers import (
    AutoConfig,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
)

from fastrag.prompters.invocation_layers.vqa_utils import get_vqa_manager

logger = logging.getLogger(__name__)


class StopWordsByTextCriteria(StoppingCriteria):
    """
    Stops text generation if any one of the stop words is generated.

    Note: When a stop word is encountered, the generation of new text is stopped.
    However, if the stop word is in the prompt itself, it can stop generating new text
    prematurely after the first token. This is particularly important for LLMs designed
    for dialogue generation. For these models, like for example mosaicml/mpt-7b-chat,
    the output includes both the new text and the original prompt. Therefore, it's important
    to make sure your prompt has no stop words.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        stop_words: List[str],
        device: Union[str, torch.device] = "cpu",
        token_backwards_buffer: int = 4,  # number of tokens to convert to a string, in addition to the stop words tokens
    ):
        super().__init__()
        self.stop_words_text = stop_words
        self.tokenizer = tokenizer
        encoded_stop_words = tokenizer(
            stop_words, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        self.stop_words = encoded_stop_words.input_ids.to(device)
        self.token_backwards_buffer = token_backwards_buffer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_word in self.stop_words:
            found_stop_word = self.is_stop_word_found(input_ids, stop_word)
            if found_stop_word:
                return True
        return False

    def is_stop_word_found(self, generated_text_ids: torch.Tensor, stop_word: torch.Tensor) -> bool:
        generated_text_ids = generated_text_ids[-1]
        len_generated_text_ids = generated_text_ids.size(0)
        len_stop_word = stop_word.size(0)
        len_stop_word_with_buffer = (
            len_stop_word + self.token_backwards_buffer
        )  # include a bit more text backwards, to ensure if the stop word indeed occurs or not

        tokens_to_decode = generated_text_ids
        if len_generated_text_ids > len_stop_word_with_buffer:
            tokens_to_decode = generated_text_ids[
                len_generated_text_ids - len_stop_word_with_buffer :
            ]

        gen_text = self.tokenizer.decode(tokens_to_decode)
        return any([sw in gen_text for sw in self.stop_words_text])


class VQAHFLocalInvocationLayer(HFLocalInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained visual question answering model from Hugging Face and
    passes a prepared prompt into that model.
    """

    def __init__(
        self,
        model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        **kwargs,
    ):
        torch_and_transformers_import.check()

        PromptModelInvocationLayer.__init__(self, model_name_or_path)
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
        for keys_to_remove in ["api_key", "task_name"]:
            if keys_to_remove in kwargs:
                del kwargs[keys_to_remove]

        pipeline_kwargs.update(kwargs)

        # create the transformer pipeline

        if "vqa_type" in pipeline_kwargs:
            vqa_type = pipeline_kwargs["vqa_type"]
        else:
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            vqa_type = model_config.architectures[0]

        vqa_manager_class = get_vqa_manager(vqa_type)

        self.vqa_manager = vqa_manager_class(
            pretrained_model_name_or_path=model_name_or_path, pipeline_kwargs=pipeline_kwargs
        )
        self.pipe = self.vqa_manager.get_pipeline()

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
        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        stream = stream or stream_handler is not None
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
            if stop_words:
                sw = StopWordsByTextCriteria(
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
                stream_handler: TokenStreamingHandler = (
                    stream_handler or DefaultTokenStreamingHandler()
                )
                model_input_kwargs["streamer"] = HFTokenStreamingHandler(
                    self.pipe.tokenizer, stream_handler
                )

            image_obj = kwargs["images"][0] if "images" in kwargs else None

            output_response = self.vqa_manager.generate(
                prompt,
                image_obj,
                model_input_kwargs=model_input_kwargs,
            )
            output = [{"generated_text": output_response}]

        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

        if stop_words:
            # Although HF generates text until stop words are encountered unfortunately it includes the stop word
            # We want to exclude it to be consistent with other invocation layers
            for idx, _ in enumerate(generated_texts):
                cur_gen_text = generated_texts[idx]
                cur_gen_text_len = len(cur_gen_text)
                for stop_word in stop_words:
                    if stop_word == cur_gen_text[cur_gen_text_len - len(stop_word) :]:
                        generated_texts[idx] = generated_texts[idx][
                            : cur_gen_text_len - len(stop_word)
                        ]
        clean_new_lines = "\n\n\n"
        clean_generated_texts = []
        for g in generated_texts:
            if clean_new_lines in g:
                while clean_new_lines in g:
                    g = g.replace(clean_new_lines, "\n")
            clean_generated_texts.append(g)

        return clean_generated_texts

    def clear_cached(self):
        self.vqa_manager.clear_cached()
