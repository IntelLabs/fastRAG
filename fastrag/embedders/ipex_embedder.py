from typing import Dict, List, Optional, Tuple, Union

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
        max_seq_length: Optional[int] = None,
        padding: Optional[bool] = True,
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

            def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
                """
                Override of original st.models.Transformer 'Tokenizes' method to add fixed length tokenization.
                """
                output = {}
                if isinstance(texts[0], str):
                    to_tokenize = [texts]
                elif isinstance(texts[0], dict):
                    to_tokenize = []
                    output["text_keys"] = []
                    for lookup in texts:
                        text_key, text = next(iter(lookup.items()))
                        to_tokenize.append(text)
                        output["text_keys"].append(text_key)
                    to_tokenize = [to_tokenize]
                else:
                    batch1, batch2 = [], []
                    for text_tuple in texts:
                        batch1.append(text_tuple[0])
                        batch2.append(text_tuple[1])
                    to_tokenize = [batch1, batch2]

                # strip
                to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

                # Lowercase
                if self.do_lower_case:
                    to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

                output.update(
                    self.tokenizer(
                        *to_tokenize,
                        padding=self.padding,
                        truncation=True,
                        return_tensors="pt",
                        max_length=self.max_seq_length,
                    )
                )
                return output

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

        if max_seq_length is not None:
            self.model._first_module().max_seq_length = max_seq_length
        self.model._first_module().padding = padding


def ipex_model_warm_up(self):
    """
    Initializes the component.
    """
    if not hasattr(self, "embedding_backend"):
        self.embedding_backend = _IPEXSentenceTransformersEmbeddingBackend(
            model=self.model,
            device=self.device.to_torch_str(),
            auth_token=self.token,
            max_seq_length=self.max_seq_length,
            padding=self.padding,
        )


class IPEXSentenceTransformersDocumentEmbedder(SentenceTransformersDocumentEmbedder):
    """
    A document embedder that uses IPEX backend for efficient computation.

    This class extends the base `SentenceTransformersDocumentEmbedder` class and provides an implementation
    that utilizes IPEX for faster document embedding computation.

    Parameters:
        max_seq_length (int, optional): The maximum sequence length of the input documents. Defaults to None.
        padding (bool or str, optional): Whether to pad the input documents to the maximum sequence length.
            If True, padding is enabled. If False, padding is disabled. If "max_length", padding is enabled
            and the input documents are padded to the maximum sequence length. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the base class constructor.
    """

    def __init__(self, max_seq_length=None, padding=True, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.padding = padding


class IPEXSentenceTransformersTextEmbedder(SentenceTransformersTextEmbedder):
    """
    A text embedder that uses IPEX backend for efficient text embedding.

    This class extends the `SentenceTransformersTextEmbedder` class and provides
    an implementation that utilizes IPEX for faster and more efficient text embedding.

    Parameters:
        max_seq_length (int, optional): The maximum sequence length of the input text. Defaults to None.
        padding (bool or str, optional): Whether to pad the input documents to the maximum sequence length.
            If True, padding is enabled. If False, padding is disabled. If "max_length", padding is enabled
            and the input documents are padded to the maximum sequence length. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    """

    def __init__(self, max_seq_length=None, padding=True, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.padding = padding


IPEXSentenceTransformersDocumentEmbedder.warm_up = ipex_model_warm_up
IPEXSentenceTransformersTextEmbedder.warm_up = ipex_model_warm_up
