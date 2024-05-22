import logging
from typing import Any, Dict, List, Optional, Union

import torch
from haystack import Document, component
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice, Secret
from haystack.utils.hf import StopWordsCriteria
from transformers import AutoTokenizer, GenerationConfig, Pipeline, StoppingCriteriaList, pipeline

from .fid_utils import FiDConverter, FiDforConditionalGeneration

logger = logging.getLogger(__name__)


@component
class FiDGenerator(HuggingFaceLocalGenerator):
    def __init__(
        self,
        model: str = "Intel/fid_flan_t5_base_nq",
        task: str = "text2text-generation",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = {},
        stop_words: Optional[List[str]] = None,
    ):
        super(FiDGenerator, self).__init__(
            model=model,
            task=task,
            device=device,
            token=token,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs.copy(),
            stop_words=stop_words,
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        self.orig_kwargs = huggingface_pipeline_kwargs
        self.huggingface_pipeline_kwargs["tokenizer"] = tokenizer
        self.input_converter = FiDConverter(tokenizer.model_max_length)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.pipeline is None:
            model_name = self.huggingface_pipeline_kwargs.pop("model")
            model_instance = FiDforConditionalGeneration.from_pretrained(
                model_name, **self.orig_kwargs
            )
            self.huggingface_pipeline_kwargs["model"] = model_instance
            super(FiDGenerator, self).warm_up()

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        documents: List[Document],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.pipeline is None:
            raise RuntimeError(
                "The generation model has not been loaded. Please call warm_up() before running."
            )

        if not prompt:
            return {"replies": []}

        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # Input conversion using FiDConverter
        inputs = self.input_converter(self.pipeline.tokenizer, prompt, documents).to(
            self.pipeline.model.device
        )

        gen_outputs = self.pipeline.model.generate(
            **inputs,
            stopping_criteria=self.stopping_criteria_list,
            **updated_generation_kwargs,
        )

        generated_tokens = gen_outputs[0]
        replies = self.pipeline.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if self.stop_words:
            # the output of the pipeline includes the stop word
            replies = [
                reply.replace(stop_word, "").rstrip()
                for reply in replies
                for stop_word in self.stop_words
            ]

        return {"replies": replies}
