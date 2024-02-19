from pathlib import Path
from typing import Optional, Union

import torch
from haystack.errors import HaystackError
from haystack.lazy_imports import LazyImport
from transformers import AutoTokenizer

from fastrag.rankers import BiEncoderRanker

with LazyImport(
    "Run pip install .[intel]' to install Intel optimized frameworks"
) as optimized_intel_import:
    from optimum.intel import IPEXModel


class QuantizedBiEncoderRanker(BiEncoderRanker):
    """
    Quantized Bi-encoder re-ranker model based on BiEncoderRanker capabilities.
    Configured to run using `optimum-intel` framework (neural compressor and Intel extension for Pytorch).

    This module only works with `optimum-intel` quantized models and using a CPU backend.
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        model_version: Optional[str] = None,
        top_k: int = 10,
        batch_size: int = 16,
        scale_score: bool = True,
        pooling: str = "mean",  # "mean" and "cls" are supported
        progress_bar: bool = True,
        query_prompt: Optional[str] = None,
        document_prompt: Optional[str] = None,
        pad_to_max: Optional[bool] = False,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'sentence-transformers/all-MiniLM-L12-v2'
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param top_k: The maximum number of documents to return
        :param batch_size: Number of documents to process at a time.
        :param scale_score: The raw predictions will be transformed using a Sigmoid activation function in case the model
                    only predicts a single label. For multi-label predictions, no scaling is applied. Set this
                    to False if you do not want any scaling of the raw predictions.
        :param progress_bar: Whether to show a progress bar while processing the documents.
        :param use_auth_token: The API token used to download private models from Huggingface.
                        If this parameter is set to `True`, then the token generated when running
                        `transformers-cli login` (stored in ~/.huggingface) will be used.
                        Additional information can be found here
                        https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param pooling: The pooling method to use to represent the encoded document or query. Supported methods: 'cls' - take the [CLS] token vector; 'mean' - mean of all the document/query vectors.
        :param query_prompt: Optional. A prefix to add to a query representation.
        :param document_prompt: Optional. A prefix to add to a document representation.
        :param pad_to_max: Optional. Pad the samples to the model's maximum number of positions instead of per batch maximum number of tokens. Recommanded to use when embedding many documents. Works only in `predict_batch`.
        """
        optimized_intel_import.check()
        super().__init__(
            model_name_or_path,
            model_version,
            top_k,
            False,
            "cpu",
            batch_size,
            scale_score,
            pooling,
            progress_bar,
            query_prompt,
            document_prompt,
            pad_to_max,
            use_auth_token,
        )

    def warm_up(self):
        try:
            self.transformer_model = IPEXModel.from_pretrained(self.model_name_or_path)
        except:
            raise HaystackError(
                f"Failed to load an optimized IPEXModel from {self.model_name_or_path}"
            )
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            revision=self.model_version,
            use_auth_token=self.use_auth_token,
        )
        self.transformer_model.eval()

    def _embed(self, inputs):
        with torch.inference_mode(), torch.cpu.amp.autocast():
            outputs = self.transformer_model(**inputs)
            if self.pooling == "mean":
                emb = self._mean_pooling(outputs, inputs["attention_mask"])
            elif self.pooling == "cls":
                emb = self._cls_pooling(outputs)
            else:
                raise ValueError("pooling method no supported")

            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb

    def __call__(self, inputs):
        return self._embed(inputs)
