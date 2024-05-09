from typing import Any, Dict, List, Optional

from haystack import component
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret

with LazyImport("Install ipex using 'pip install -e .[intel]'") as ipex_import:
    import intel_extension_for_pytorch as ipex
    import torch
    from haystack.utils.hf import StopWordsCriteria  # pylint: disable=ungrouped-imports
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList


class IPEXGenerator(HuggingFaceLocalGenerator):
    """
    A generator class that uses IPEX (Intel PyTorch Extension) for faster inference on Xeon CPUs.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model. Defaults to "meta-llama/Llama-2-7b-hf".
        device (Optional[ComponentDevice]): The device to run the generator on. Defaults to None.
        token (Optional[Secret]): The secret token for authentication. Defaults to Secret.from_env_var("HF_API_TOKEN", strict=False).
        generation_kwargs (Optional[Dict[str, Any]]): Additional generation arguments. Defaults to None.
        huggingface_pipeline_kwargs (Optional[Dict[str, Any]]): Additional arguments for the Hugging Face pipeline. Defaults to None.
        stop_words (Optional[List[str]]): A list of stop words to remove from the generated replies. Defaults to None.
        dtype (Optional[str]): The data type to use for inference. Defaults to "bfloat16".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        dtype: Optional[str] = "bfloat16",  ## bfloat16 of float32
        **kwargs,
    ):
        ipex_import.check()
        super().__init__(
            model=model_name_or_path,
            device=device,
            token=token,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs,
            stop_words=stop_words,
        )
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = token
        self.stop_words = stop_words
        self.generation_kwargs = generation_kwargs or {}
        self.amp_enabled = True if dtype != "float32" else False
        self.amp_dtype = getattr(torch, dtype)

    def warm_up(self):
        """
        Initializes the generator by loading the pre-trained model and tokenizer.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.amp_dtype,
            low_cpu_mem_usage=True,
            token=self.token,
        ).eval()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model = ipex.llm.optimize(
            self.model,
            dtype=self.amp_dtype,
            inplace=True,
            deployment_mode=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=self.token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.huggingface_pipeline_kwargs["tokenizer"] = self.tokenizer
        if self.stop_words and self.stopping_criteria_list is None:
            stop_words_criteria = StopWordsCriteria(
                tokenizer=self.pipeline.tokenizer,
                stop_words=self.stop_words,
                device=self.pipeline.device,
            )
            self.stopping_criteria_list = StoppingCriteriaList([stop_words_criteria])

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generates replies based on the given prompt.

        Args:
            prompt (str): The prompt for generating replies.
            generation_kwargs (Optional[Dict[str, Any]]): Additional generation arguments. Defaults to None.

        Returns:
            Dict[str, List[str]]: A dictionary containing the generated replies.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Generator is not initialized. Run warm_up() first.")
        if not prompt:
            return {"replies": []}

        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        with torch.inference_mode(), torch.cpu.amp.autocast(enabled=self.amp_enabled):
            inputs = self.tokenizer(prompt, truncation=True, return_tensors="pt")
            gen_ids = self.model.generate(
                **inputs, stopping_criteria=self.stopping_criteria_list, **updated_generation_kwargs
            )
            generated_tokens = gen_ids[0][inputs["input_ids"].shape[1] :]
            replies = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if self.stop_words:
            for stop_word in self.stop_words:
                replies = replies.replace(stop_word, "").rstrip()
        return {"replies": [replies]}
