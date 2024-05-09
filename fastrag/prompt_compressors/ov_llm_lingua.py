import json
import os
from typing import Any

from haystack.lazy_imports import LazyImport

from fastrag.prompt_compressors.llm_lingua import LLMLinguaPromptCompressor

with LazyImport("Install openvino using 'pip install -e .[openvino]'") as ov_import:
    from optimum.intel import OVModelForTokenClassification


class OVLLMLinguaPromptCompressor(LLMLinguaPromptCompressor):
    """
    OVLLMLinguaPromptCompressor is a component that compresses the prompt it receives, while preserving the most essential
    information within it.
    For more information, we refer to the original repo: https://github.com/microsoft/LLMLingua

    Usage example:
    ```python
    compressor = OVLLMLinguaPromptCompressor()
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
        ov_import.check()
        model_config = json.load(open(os.path.join(model_name_or_path, "config.json")))
        super().__init__(
            model_name_or_path=model_config["_name_or_path"],
            device_map=device_map,
            rate=rate,
            force_tokens=force_tokens,
            target_tokens=target_tokens,
        )
        self.ov_model_name_or_path = model_name_or_path

    def warm_up(self):
        """
        Initializes the component.
        """
        super().warm_up()
        self.llm_lingua.model = OVModelForTokenClassification.from_pretrained(
            self.ov_model_name_or_path
        )
