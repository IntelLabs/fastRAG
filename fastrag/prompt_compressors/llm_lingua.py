from typing import Any

from haystack import component
from haystack.lazy_imports import LazyImport

with LazyImport("Install llmlingua using 'pip install -e .[llmlingua]'") as llmlingua_import:
    from llmlingua import PromptCompressor


@component
class LLMLinguaPromptCompressor:
    """
    LLMLinguaPromptCompressor is a component that compresses the prompt it receives, while preserving the most essential
    information within it.
    For more information, we refer to the original repo: https://github.com/microsoft/LLMLingua

    Usage example:
    ```python
    compressor = LLMLinguaPromptCompressor()
    ```
    """

    def __init__(
        self,
        model_name_or_path: Any = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map="cpu",
        rate=0.33,
        force_tokens=["\n", "?"],
        target_tokens=None,
    ):
        """
        Constructs a LLMLinguaPromptCompressor component.

        """
        llmlingua_import.check()
        self.rate = rate
        self.force_tokens = force_tokens
        self.target_tokens = target_tokens
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        component.set_input_type(self, "prompt", Any, "")

    def warm_up(self):
        """
        Initializes the component.
        """
        self.llm_lingua = PromptCompressor(
            model_name=self.model_name_or_path,
            use_llmlingua2=True,
            device_map=self.device_map,
        )

    @component.output_types(prompt=str)
    def run(self, prompt: str, **kwargs):
        """
        :param kwargs:
            The variables that will be used to render the prompt template.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the prompt template.
        """
        compressed_prompt = self.llm_lingua.compress_prompt(
            prompt, rate=self.rate, force_tokens=self.force_tokens
        )

        return {"prompt": compressed_prompt["compressed_prompt"]}
