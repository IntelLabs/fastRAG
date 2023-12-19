import logging
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import transformers
from haystack import Document
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
from torch import nn
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    Pipeline,
    StoppingCriteriaList,
    pipeline,
)
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleOutput,
)

logger = logging.getLogger(__name__)


class REPLUG_Generation(GenerationMixin):
    """Implementing REPLUG-based sampling text generation."""

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        )
        output_scores = (
            output_scores if output_scores is not None else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(
                    input_ids.device
                )
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            ### REPLUG - document weighting is done via REPLUGLogitsProcessor
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            ### REPLUG
            # Sample from the normalized "logits", assuming the REPLUG processor was used!
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Lock same next-token for all examples in batch
            next_tokens[:] = next_tokens[0]

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


class REPLUGLogitsProcessor(LogitsProcessor):
    """
    REPLUG processor uses documents similarity scores to weight the different examples in a batch.
    """

    def __init__(self, documents_scores: torch.FloatTensor):
        "Documents' scores assumed to be L1 normalized."
        self.num_docs = documents_scores.shape[0]
        self.doc_scores = torch.unsqueeze(documents_scores, 1)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        "Scores assumed to be logits. Returns logits averaged over the batch."
        replug_scores = self.doc_scores * scores
        replug_scores = replug_scores.sum(dim=0)
        replug_scores = torch.tile(replug_scores, (self.num_docs, 1))
        return replug_scores


class HF_REPLUG:
    "Creates a HF model that inherits from REPLUG_Generation class"

    def __new__(cls, name_or_path, **kwargs):
        return factory(name_or_path).from_pretrained(name_or_path, **kwargs)


def factory(name_or_path):
    loadedConfig = AutoConfig.from_pretrained(name_or_path)
    try:
        pretrained_class_object = getattr(transformers, loadedConfig.architectures[0])
        if pretrained_class_object not in MODEL_FOR_CAUSAL_LM_MAPPING.values():
            raise ValueError(
                f"Model {pretrained_class_object} is not used for causal LM generation."
            )
    except AttributeError:
        raise ValueError("Transformers architecture unknown.")

    class HF(pretrained_class_object, REPLUG_Generation):
        """Wrapper around HuggingFace transformers with REPLUG generation."""

        _keys_to_ignore_on_load_unexpected = [r"cls"]
        _tied_weights_keys = ["lm_head.weight"]

    return HF


class ReplugHFLocalInvocationLayer(HFLocalInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained model from Hugging Face, and uses
     utilizes the model according to the REPLUG algorithm (https://arxiv.org/abs/2301.12652)
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        doc_sep_token: str = "###REPLUG-DOC###",
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

        ### REPLUG - model loading
        loading_model_kwargs = kwargs.get("model_kwargs", {})
        kwargs["model"] = HF_REPLUG(model_name_or_path, **loading_model_kwargs)

        pipeline_kwargs = self._prepare_pipeline_kwargs(
            task=self.task_name,
            model_name_or_path=model_name_or_path,
            use_auth_token=use_auth_token,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        pipeline_kwargs["tokenizer"] = tokenizer
        del pipeline_kwargs["device"]
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

        self.doc_sep_token = doc_sep_token

    def create_doc_batch_prompt(self, prompt: str, documents: List[Document]):
        "Creating REPLUG prompts using a single prompt and a list of haystack documents."
        prompts = [prompt.replace(self.doc_sep_token, doc.content) for doc in documents]
        return prompts

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
            documents = kwargs.pop("documents")
            prompts = self.create_doc_batch_prompt(prompt, documents)

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
                model_input_kwargs["max_length"] = model_input_kwargs.pop(
                    "max_length", self.max_length
                )

            if stream:
                stream_handler: TokenStreamingHandler = (
                    stream_handler or DefaultTokenStreamingHandler()
                )
                model_input_kwargs["streamer"] = HFTokenStreamingHandler(
                    self.pipe.tokenizer, stream_handler
                )

            ### REPLUG - retrieved documents processing
            scores_soft = torch.tensor([doc.score for doc in documents], dtype=torch.float32).to(
                self.pipe.model.device
            )
            scores_soft /= scores_soft.sum()

            inputs = self.pipe.tokenizer(
                prompts, truncation=True, padding=True, return_tensors="pt"
            ).to(self.pipe.model.device)

            model_input_kwargs = {
                k: v for k, v in model_input_kwargs.items() if k not in ["return_full_text"]
            }

            ### REPLUG
            gen_outputs = self.pipe.model.generate(
                **inputs,
                logits_processor=LogitsProcessorList([REPLUGLogitsProcessor(scores_soft)]),
                **model_input_kwargs,
            )

            generated_tokens = gen_outputs[0][inputs["input_ids"].shape[1] :]
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
