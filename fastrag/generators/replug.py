import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import transformers
from haystack import Document, component
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice
from torch import nn
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
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


@component
class ReplugGenerator(HuggingFaceLocalGenerator):
    """
    A subclass of the HuggingFaceLocalGenerator class. It loads a pre-trained model from Hugging Face, and uses
    utilizes the model according to the REPLUG algorithm (https://arxiv.org/abs/2301.12652)
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        device: Optional[ComponentDevice] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = {},
        doc_sep_token: str = "###REPLUG-DOC###",
        **kwargs,
    ):
        super(ReplugGenerator, self).__init__(
            model=model,
            device=device,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs.copy(),
            **kwargs,
        )

        self.model = model
        self.generation_kwargs.pop("return_full_text", None)
        self.orig_kwargs = huggingface_pipeline_kwargs

        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.huggingface_pipeline_kwargs["tokenizer"] = tokenizer

        self.doc_sep_token = doc_sep_token

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.pipeline is None:
            model_instance = HF_REPLUG(self.model, **self.orig_kwargs)
            self.huggingface_pipeline_kwargs["model"] = model_instance
            super(ReplugGenerator, self).warm_up()

    def create_doc_batch_prompt(self, prompt: str, documents: List[Document]):
        "Creating REPLUG prompts using a single prompt and a list of haystack documents."
        prompts = [prompt.replace(self.doc_sep_token, doc.content) for doc in documents]
        return prompts

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        documents: List[Document],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run the text generation model on the given prompt.

        :param prompt:
            A string representing the prompt.
        :param generation_kwargs:
            Additional keyword arguments for text generation.

        :returns:
            A dictionary containing the generated replies.
            - replies: A list of strings representing the generated replies.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "The generation model has not been loaded. Please call warm_up() before running."
            )

        if not prompt:
            return {"replies": []}

        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        ### REPLUG - retrieved documents processing
        prompts = self.create_doc_batch_prompt(prompt, documents)

        scores_norm = torch.tensor([doc.score for doc in documents], dtype=torch.float32).to(
            self.pipeline.model.device
        )
        scores_norm /= scores_norm.sum()  # L1 normalization

        inputs = self.pipeline.tokenizer(
            prompts, truncation=True, padding=True, return_tensors="pt"
        ).to(self.pipeline.model.device)

        gen_outputs = self.pipeline.model.generate(
            **inputs,
            stopping_criteria=self.stopping_criteria_list,
            logits_processor=LogitsProcessorList([REPLUGLogitsProcessor(scores_norm)]),
            **updated_generation_kwargs,
        )

        generated_tokens = gen_outputs[0][inputs["input_ids"].shape[1] :]
        replies = self.pipeline.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if self.stop_words:
            # the output of the pipeline includes the stop word
            replies = [
                reply.replace(stop_word, "").rstrip()
                for reply in replies
                for stop_word in self.stop_words
            ]

        return {"replies": replies}
