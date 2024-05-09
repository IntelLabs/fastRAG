from typing import List, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, StoppingCriteria


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
        return any([gen_text.endswith(sw) for sw in self.stop_words_text])
