import logging
import sys
from typing import Dict, List, Optional, Union

from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.hugging_face import HFLocalInvocationLayer

with LazyImport("Install llama_cpp using 'pip install -e .[llama_cpp]'") as llama_cpp_import:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCPPInvocationLayer(HFLocalInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained model from Hugging Face,
    and loads it into an HPU device, including ad-hoc optimizations.
    """

    def __init__(
        self,
        model_name_or_path: str = "llama-model.gguf",
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        PromptModelInvocationLayer.__init__(self, model_name_or_path)

        self.llm = Llama(model_path=model_name_or_path)
        self.max_length = max_length
        self.max_new_tokens = kwargs.get("max_new_tokens", 100)

        # Additional properties for Invocation Layer requirements
        self.model_max_length = kwargs.get("model_max_length", sys.maxsize)
        self.generation_kwargs = kwargs

    def _ensure_token_limit(
        self, prompt: Union[str, List[Dict[str, str]]]
    ) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        model_max_length = self.model_max_length
        tokenized_prompt = self.llm.tokenize(bytes(prompt, "utf-8"))
        n_prompt_tokens = len(tokenized_prompt)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= model_max_length:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
            "Shorten the prompt to prevent it from being cut off",
            n_prompt_tokens,
            max(0, model_max_length - n_answer_tokens),
            n_answer_tokens,
            model_max_length,
        )

        decoded_string = self.llm.detokenize(
            tokenized_prompt[: model_max_length - n_answer_tokens]
        ).decode("utf-8")
        return decoded_string

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated texts using the local Hugging Face transformers model
        :return: A list of generated texts.

        Note: Only kwargs relevant to Text2TextGenerationPipeline and TextGenerationPipeline are passed to
        Hugging Face as model_input_kwargs. Other kwargs are ignored.
        """
        output: List[Dict[str, str]] = []
        stop_words = kwargs.pop("stop_words", [])

        generated_texts = []
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            generation_kwargs = self.generation_kwargs
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

            generation_kwargs.update(model_input_kwargs)
            model_input_kwargs = generation_kwargs

            echo = model_input_kwargs.get("return_full_text", False)
            max_tokens = model_input_kwargs.get("max_new_tokens", self.max_new_tokens)

            output = self.llm(
                prompt,  # Prompt
                max_tokens=max_tokens,  # Generate up to 32 tokens
                stop=stop_words,  # Stop generating just before the model would generate a new question
                echo=echo,  # Echo the prompt back in the output
            )  # Generate a completion, can also call create_completion

            generated_texts = [output["choices"][0]["text"]]

        return generated_texts
