from typing import Optional, Union

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackend,
)
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret

with LazyImport(
    "Run pip install .[intel]' to install Intel optimized frameworks"
) as optimized_intel_import:
    from optimum.intel import IPEXModel


class _IPEXSentenceTransformersEmbeddingBackend(_SentenceTransformersEmbeddingBackend):
    """
    Class to manage Sentence Transformers embeddings.
    """

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
    ):
        import sentence_transformers

        class _IPEXSTTransformers(sentence_transformers.models.Transformer):
            def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
                print("Loading IPEX ST Transformer model")
                optimized_intel_import.check()
                self.auto_model = IPEXModel.from_pretrained(
                    model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                )
                self.auto_model.eval()

        class _IPEXSentenceTransformer(sentence_transformers.SentenceTransformer):
            def _load_auto_model(
                self,
                model_name_or_path: str,
                token: Optional[Union[bool, str]],
                cache_folder: Optional[str],
                revision: Optional[str] = None,
                trust_remote_code: bool = False,
            ):
                """
                Creates a simple Transformer + Mean Pooling model and returns the modules
                """
                transformer_model = _IPEXSTTransformers(
                    model_name_or_path,
                    cache_dir=cache_folder,
                    model_args={
                        "token": token,
                        "trust_remote_code": trust_remote_code,
                        "revision": revision,
                    },
                    tokenizer_args={
                        "token": token,
                        "trust_remote_code": trust_remote_code,
                        "revision": revision,
                    },
                )
                pooling_model = sentence_transformers.models.Pooling(
                    transformer_model.get_word_embedding_dimension(), "mean"
                )
                return [transformer_model, pooling_model]

            @property
            def device(self):
                return "cpu"

        self.model = _IPEXSentenceTransformer(
            model_name_or_path=model,
            device=device,
            use_auth_token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
        )


def ipex_model_warm_up(self):
    """
    Initializes the component.
    """
    if not hasattr(self, "embedding_backend"):
        self.embedding_backend = _IPEXSentenceTransformersEmbeddingBackend(
            model=self.model,
            device=self.device.to_torch_str(),
            auth_token=self.token,
        )


class IPEXSentenceTransformersDocumentEmbedder(SentenceTransformersDocumentEmbedder):
    """
    A document embedder that uses IPEX for efficient computation.

    This class extends the base `SentenceTransformersDocumentEmbedder` class and provides an implementation
    that utilizes IPEX for faster document embedding computation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IPEXSentenceTransformersTextEmbedder(SentenceTransformersTextEmbedder):
    """
    A text embedder that uses IPEX for efficient text embedding.

    This class extends the `SentenceTransformersTextEmbedder` class and provides
    an implementation that utilizes IPEX for faster and more efficient text embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


IPEXSentenceTransformersDocumentEmbedder.warm_up = ipex_model_warm_up
IPEXSentenceTransformersTextEmbedder.warm_up = ipex_model_warm_up
