import copy
import logging
from typing import List

import torch
import torch.nn.functional as F
from haystack import Document
from torch import nn
from transformers import BatchEncoding
from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration, T5Stack

logger = logging.getLogger(__name__)


def get_padded_tensor(ten_list, value=0):
    max_len = max([x.shape[1] for x in ten_list])
    padded_list = []
    for tensor in ten_list:
        if tensor.shape[1] < max_len:
            tensor = F.pad(
                input=tensor, pad=(0, max_len - tensor.shape[1], 0, 0), mode="constant", value=value
            )
        padded_list.append(tensor)
    return padded_list


def tokenization_result_to_tensor(inputs):
    inputs = get_padded_tensor(inputs, value=0)
    return torch.stack(inputs)


def passages_to_tensors(tokenizer, passage_batch, passage_max_len, pad_to_max_length):
    # if padding to the max length, no padding to max passage length, and vice versa.
    padding = True if not pad_to_max_length else "max_length"

    all_input_ids = []
    all_masks = []

    for input_passages in passage_batch:
        current_passage_batch_result = tokenizer(
            input_passages,
            max_length=passage_max_len,
            truncation=True,
            add_special_tokens=False,
            padding=padding,
            return_tensors="pt",
        )

        all_input_ids.append(current_passage_batch_result["input_ids"])
        all_masks.append(current_passage_batch_result["attention_mask"])

    all_input_ids = tokenization_result_to_tensor(all_input_ids)
    all_masks = tokenization_result_to_tensor(all_masks).bool()

    return all_input_ids, all_masks


class FiDConverter:
    def __init__(self, tokenizer_max_len):
        self.max_len = tokenizer_max_len
        logger.info(f"tokenizer max length is:{self.max_len}")
        self.pre_context = "context:"
        self.pre_title = "title:"
        self.pre_question = "question:"

    def __call__(self, tokenizer, query: str, documents: List[Document], top_k=None):
        model_max_len = tokenizer.model_max_length
        if model_max_len > self.max_len:
            logger.info(
                f"Warning!!! You set {self.max_len} as max len for the tokenizer, which is smaller than model_max_length {model_max_len}."
            )

        question = self.pre_question + " " + query

        first_doc = documents[0]
        if first_doc.meta.get("title") is not None:
            formatted_passages = [
                f"{self.pre_title} {c.meta['title']} {self.pre_context} {c.content}"
                for c in documents
            ]
        else:
            formatted_passages = [f"{self.pre_context} {c.content}" for c in documents]

        formatted_passages_with_question = [[question + " " + t for t in formatted_passages]]
        all_input_ids, all_masks = passages_to_tensors(
            tokenizer, formatted_passages_with_question, self.max_len, False
        )
        return BatchEncoding({"input_ids": all_input_ids, "attention_mask": all_masks})


class FusionInDecoderStack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        # init the block stack using the shared embed_tokens
        super().__init__(config, embed_tokens=embed_tokens)
        self.is_encoder = not self.is_decoder

    def check_for_encoder_input_preprocessing(self, input_ids, attention_mask):
        batch_size = None
        if self.is_encoder:
            input_ids_shape = input_ids.shape
            assert len(input_ids_shape) == 3
            passage_count = input_ids_shape[1]
            batch_size = input_ids_shape[0]

            # flatten batch_size and passage dimensions
            input_ids = input_ids.view(batch_size * passage_count, -1)
            attention_mask = attention_mask.view(batch_size * passage_count, -1)

        return input_ids, attention_mask, batch_size

    def get_last_hidden_state(self, output, return_dict):
        return output.last_hidden_state if return_dict else output[0]

    def output_last_hidden_state(self, output, last_hidden_state, return_dict):
        if return_dict:
            output.last_hidden_state = last_hidden_state
            return output
        else:
            return tuple(last_hidden_state, *output[1:])

    def check_for_encoder_output_preprocessing(self, output, return_dict, batch_size):
        if self.is_encoder:
            last_hidden_state = self.get_last_hidden_state(output, return_dict)
            last_hidden_state = last_hidden_state.view(batch_size, -1, last_hidden_state.shape[-1])
            return self.output_last_hidden_state(output, last_hidden_state, return_dict)
        return output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.gradient_checkpointing and self.training:
            use_cache = False

        input_ids, attention_mask, batch_size = self.check_for_encoder_input_preprocessing(
            input_ids, attention_mask
        )

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_attention_mask = encoder_attention_mask.view(*encoder_hidden_states.shape[:-1])

        forward_result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self.check_for_encoder_output_preprocessing(forward_result, return_dict, batch_size)


class FiDforConditionalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        # we utilize the same init function as the original T5ForConditionalGeneration,
        # except for using our FusionInDecoderStack
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FusionInDecoderStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FusionInDecoderStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
