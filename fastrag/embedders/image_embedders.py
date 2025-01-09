import base64
from io import BytesIO
from typing import List

import torch
from haystack import Document, component
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def base64_to_image(base_64_input):
    bytes_io = BytesIO(base64.b64decode(base_64_input))
    return Image.open(bytes_io)


class BaseSentenceTransformersImageEmbedder(SentenceTransformersDocumentEmbedder):
    def warm_up(self):
        """
        Initializes the component.
        """

        # Load CLIP model
        self.embeding_backend = CLIPModel.from_pretrained(self.model)
        self.processor = CLIPProcessor.from_pretrained(self.model)


@component
class SentenceTransformersImageEmbedder(
    BaseSentenceTransformersImageEmbedder, SentenceTransformersDocumentEmbedder
):
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        images = [base64_to_image(doc.meta["image_base64"]) for doc in documents]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.embeding_backend.get_image_features(**inputs)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}


@component
class SentenceTransformersImageTextEmbedder(
    BaseSentenceTransformersImageEmbedder, SentenceTransformersTextEmbedder
):
    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        Embed a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_outputs = self.embeding_backend.get_text_features(**inputs)

        return {"embedding": text_outputs.tolist()[0]}
