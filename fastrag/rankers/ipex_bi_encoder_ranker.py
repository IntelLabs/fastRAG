from fastrag.embedders import IPEXSentenceTransformersDocumentEmbedder
from fastrag.rankers.bi_encoder_ranker import BiEncoderSimilarityRanker


class IPEXBiEncoderSimilarityRanker(BiEncoderSimilarityRanker):
    """
    A similarity ranker that uses a quantized bi-encoder model for document embeddings.

    Inherits from the BiEncoderSimilarityRanker class.

    Attributes:
        embedder_model (IPEXSentenceTransformersDocumentEmbedder): The quantized bi-encoder document embedder model.
    """

    def warm_up(self):
        self.embedder_model = IPEXSentenceTransformersDocumentEmbedder(
            model=self.model_name_or_path,
            device=self.device,
            token=self.token,
            prefix=self.document_prefix,  ## we'll use only for document embeddings, query is one time
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )
        self.embedder_model.warm_up()
