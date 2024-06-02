import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch
from haystack import component
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret

with LazyImport("Install optimum-habana using 'pip install -e .[habana]'") as gaudi_import:
    from fastrag.generators.gaudi_utils import adjust_batch, initialize_model

logger = logging.getLogger()


def setup_parser(huggingface_pipeline_kwargs: dict, model_name_or_path: str):
    """
    Optimum-habana text generation paramters.
    """
    huggingface_pipeline_kwargs.setdefault(
        "static_shapes", True
    )  # whether to use static shapes, if possible
    huggingface_pipeline_kwargs.setdefault(
        "input_padding", False
    )  # whether to apply padding to the input_ids when max_input_tokens == 0

    #   For a detailed description of each parameter, please visit:
    #   https://github.com/huggingface/optimum-habana/blob/main/examples/text-generation/run_generation.py
    #   All settings are explained in the setup_parser function.

    huggingface_pipeline_kwargs.setdefault("device", "hpu")
    huggingface_pipeline_kwargs.setdefault("model_name_or_path", model_name_or_path)
    huggingface_pipeline_kwargs.setdefault("bf16", True)
    huggingface_pipeline_kwargs.setdefault("max_new_tokens", 100)
    huggingface_pipeline_kwargs.setdefault("max_input_tokens", 0)
    huggingface_pipeline_kwargs.setdefault("batch_size", 1)
    huggingface_pipeline_kwargs.setdefault("warmup", 3)
    huggingface_pipeline_kwargs.setdefault("n_iterations", 5)
    huggingface_pipeline_kwargs.setdefault("local_rank", 0)
    huggingface_pipeline_kwargs.setdefault("use_kv_cache", True)
    huggingface_pipeline_kwargs.setdefault("use_hpu_graphs", True)
    huggingface_pipeline_kwargs.setdefault("dataset_name", None)
    huggingface_pipeline_kwargs.setdefault("column_name", None)
    huggingface_pipeline_kwargs.setdefault("do_sample", True)
    huggingface_pipeline_kwargs.setdefault("num_beams", 1)
    huggingface_pipeline_kwargs.setdefault("trim_logits", True)
    huggingface_pipeline_kwargs.setdefault("seed", 27)
    huggingface_pipeline_kwargs.setdefault("profiling_warmup_steps", 0)
    huggingface_pipeline_kwargs.setdefault("profiling_steps", 0)
    huggingface_pipeline_kwargs.setdefault("prompt", None)
    huggingface_pipeline_kwargs.setdefault("bad_words", None)
    huggingface_pipeline_kwargs.setdefault("force_words", None)
    huggingface_pipeline_kwargs.setdefault("peft_model", None)
    huggingface_pipeline_kwargs.setdefault("num_return_sequences", 1)
    huggingface_pipeline_kwargs.setdefault("token", None)
    huggingface_pipeline_kwargs.setdefault("model_revision", "main")
    huggingface_pipeline_kwargs.setdefault("attn_softmax_bf16", True)
    huggingface_pipeline_kwargs.setdefault("output_dir", None)
    huggingface_pipeline_kwargs.setdefault("bucket_size", -1)
    huggingface_pipeline_kwargs.setdefault(
        "bucket_internal", False
    )  # set to False to work for the default case bucket_size = -1
    huggingface_pipeline_kwargs.setdefault("dataset_max_samples", -1)
    huggingface_pipeline_kwargs.setdefault("limit_hpu_graphs", True)
    huggingface_pipeline_kwargs.setdefault("reuse_cache", True)
    huggingface_pipeline_kwargs.setdefault("verbose_workers", True)
    huggingface_pipeline_kwargs.setdefault("simulate_dyn_prompt", None)
    huggingface_pipeline_kwargs.setdefault("reduce_recompile", False)
    huggingface_pipeline_kwargs.setdefault("fp8", True)
    huggingface_pipeline_kwargs.setdefault("use_flash_attention", True)
    huggingface_pipeline_kwargs.setdefault("flash_attention_recompute", True)
    huggingface_pipeline_kwargs.setdefault("flash_attention_causal_mask", True)
    huggingface_pipeline_kwargs.setdefault("book_source", True)
    huggingface_pipeline_kwargs.setdefault("torch_compile", True)
    huggingface_pipeline_kwargs.setdefault("ignore_eos", True)
    huggingface_pipeline_kwargs.setdefault("temperature", 1.0)
    huggingface_pipeline_kwargs.setdefault("top_p", 1.0)
    huggingface_pipeline_kwargs.setdefault("const_serialization_path", None)
    huggingface_pipeline_kwargs.setdefault("disk_offload", True)
    huggingface_pipeline_kwargs.setdefault("trust_remote_code", True)

    args = Namespace()
    for k, v in huggingface_pipeline_kwargs.items():
        setattr(args, k, v)

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config == "" and args.disk_offload:
        logger.info(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )
    return args


class GaudiGenerator(HuggingFaceLocalGenerator):
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
        model_name_or_path: str = "microsoft/Phi-3-mini-128k-instruct",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        dtype: Optional[str] = "bfloat16",  ## bfloat16 of float32
        **kwargs,
    ):
        gaudi_import.check()
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
        self.token = token
        self.stop_words = stop_words
        self.generation_kwargs = generation_kwargs or {}

    def warm_up(self):
        """
        Initializes the generator by loading the pre-trained model and tokenizer.
        """
        if self.model is None:
            self.args = setup_parser(self.huggingface_pipeline_kwargs, self.model_name_or_path)
            self.model, self.tokenizer, self.generation_config = initialize_model(self.args, logger)
            self.tokenizer.padding_side = "left"
            self.use_lazy_mode = True

            if self.args.torch_compile and self.model.config.model_type == "llama":
                self.use_lazy_mode = False

            import habana_frameworks.torch.hpu as torch_hpu

    def generate(self, input_sentences):
        """Generates sequences from the input sentences and returns them."""
        size = None

        encode_t0 = time.perf_counter()
        # Tokenization
        if self.args.max_input_tokens > 0:
            input_tokens = self.tokenizer.batch_encode_plus(
                input_sentences,
                return_tensors="pt",
                padding="max_length",
                max_length=self.args.max_input_tokens,
                truncation=True,
            )
        else:
            input_tokens = self.tokenizer.batch_encode_plus(
                input_sentences, return_tensors="pt", padding=self.args.input_padding
            )

        encode_duration = time.perf_counter() - encode_t0

        if size is not None:
            input_tokens = adjust_batch(input_tokens, size)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.args.device)
        iteration_times = []
        generation_kwargs = {
            k: v for k, v in self.generation_kwargs.items() if k not in ["return_full_text"]
        }
        outputs = self.model.generate(
            **input_tokens,
            generation_config=self.generation_config,
            lazy_mode=self.use_lazy_mode,
            hpu_graphs=self.args.use_hpu_graphs,
            profiling_steps=self.args.profiling_steps,
            profiling_warmup_steps=self.args.profiling_warmup_steps,
            ignore_eos=self.args.ignore_eos,
            iteration_times=iteration_times,
            **generation_kwargs,
        ).cpu()
        first_token_time = iteration_times[0] + encode_duration
        logger.info(f"Time to first token = {first_token_time*1000}ms")
        outputs = outputs[:, input_tokens["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
        replies = self.generate([prompt])
        return {"replies": replies}
