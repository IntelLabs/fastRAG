import base64
from io import BytesIO
from itertools import product
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.answer_generator.base import BaseGenerator


def pil_to_base64(obj):
    buffered = BytesIO()
    obj.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def get_answer_text(obj):
    return obj.answer


def get_title_text(obj):
    return obj.meta["title"]


def get_content_text(obj):
    return obj.content


def get_query_text(obj):
    return obj


def merge_text(texts):
    return ". ".join(texts)


def get_full_document_text(obj):
    return merge_text(
        [
            get_title_text(obj),
            get_content_text(obj),
        ]
    )


GET_TEXT_MAP = {
    "query": get_query_text,
    "title": get_title_text,
    "content": get_content_text,
    "full_document": get_full_document_text,
    "answer": get_answer_text,
}

DEFAULT_GET_TEXT = "answer"

GET_TEXT_LOCATION = {
    "query": "query",
    "title": "documents",
    "content": "documents",
    "full_document": "documents",
    "answer": "answers",
}


class ImageDiffuserGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        access_token: str = None,
        use_gpu: bool = True,
        batch_size: int = 1,
        num_inference_steps: int = 15,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        super().__init__(progress_bar=progress_bar)

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
            revision="fp16" if use_gpu else "fp32",
            use_auth_token=access_token,
        )
        self.devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )

        self.pipe = pipe.to(str(self.devices[0]))
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps

    def get_image(self, texts, num_inference_steps: int = -1):
        # use default num_inference_steps if not provided
        num_inference_steps = (
            num_inference_steps if num_inference_steps != -1 else self.num_inference_steps
        )
        images = self.pipe(texts, num_inference_steps=num_inference_steps).images
        return [pil_to_base64(image) for image in images]

    def get_text_combination_objects(self, query_dict, generate_from):
        """
        example:
        generate_from = 'query,answer,title,content'
        query_dict = {
            "query": "a",
            "answers": [Namespace(answer="b",meta=Namespace()), Namespace(answer="c",meta=Namespace())],
            "documents": [
                Namespace(content="d",meta={"title": "e"}),
                Namespace(content="f",meta={"title": "g"}),
            ]
        }
        """
        generate_from_parts = generate_from.split(",")

        generate_locations = set([GET_TEXT_LOCATION[part] for part in generate_from_parts])

        text_combinations = list(product(*[query_dict[part] for part in generate_locations]))

        merged_texts = []
        for comb in text_combinations:
            text_mapping = {k: comb[i] for i, k in enumerate(generate_locations)}
            texts = []
            objs = []

            for part in generate_from_parts:
                text_mapping_key = GET_TEXT_LOCATION[part]
                obj = text_mapping[text_mapping_key]
                text = GET_TEXT_MAP[part](obj)
                texts.append(text)
                objs.append(obj)

            merged_text = merge_text(texts)
            merged_texts.append({"merged_text": merged_text, "objs": objs})
        return merged_texts

    def get_all_merged_texts(self, query_dict, generate_from_method):
        merged_texts = []
        for q, a, d in zip(query_dict["query"], query_dict["answers"], query_dict["documents"]):
            query_dict_current = dict(
                query=q,
                answers=a,
                documents=d,
            )
            merged_texts += self.get_text_combination_objects(
                query_dict_current, generate_from_method
            )
        return merged_texts

    def predict_batch(
        self,
        queries=None,
        answers=None,
        documents=None,
        generate_from: str = [DEFAULT_GET_TEXT],
        num_inference_steps: int = -1,
    ):
        query_dict = dict(query=queries, answers=answers, documents=documents, images={})

        for generate_from_method in generate_from:
            merged_texts = self.get_all_merged_texts(query_dict, generate_from_method)

            chunks = (len(merged_texts) // self.batch_size) + 1
            for chunk_index in range(chunks):
                start_index = chunk_index * self.batch_size
                end_index = (chunk_index + 1) * self.batch_size

                batch = merged_texts[start_index:end_index]
                if len(batch) == 0:
                    continue

                images = self.get_image([t["merged_text"] for t in batch], num_inference_steps)
                batch_index_range = list(range(start_index, end_index))

                for image_index, image in enumerate(images):
                    merged_text_index = batch_index_range[image_index]
                    merged_texts[merged_text_index]["image"] = image

            query_dict["images"][generate_from_method] = merged_texts

        return query_dict

    def predict(
        self,
        query=None,
        answers=None,
        documents=None,
        generate_from: str = [DEFAULT_GET_TEXT],
        num_inference_steps: int = -1,
    ):
        predicted = self.predict_batch(
            queries=[query],
            answers=[answers],
            documents=[documents],
            generate_from=generate_from,
            num_inference_steps=num_inference_steps,
        )
        for key in ["query", "answers", "documents"]:
            predicted[key] = predicted[key][0]

        return predicted

    def run(
        self,
        query=None,
        answers=None,
        documents=None,
        generate_from: str = [DEFAULT_GET_TEXT],
        num_inference_steps: int = -1,
    ):
        return (
            self.predict(
                query=query,
                answers=answers,
                documents=documents,
                generate_from=generate_from,
                num_inference_steps=num_inference_steps,
            ),
            "output1",
        )

    def run_batch(
        self,
        queries=None,
        answers=None,
        documents=None,
        generate_from: str = [DEFAULT_GET_TEXT],
        num_inference_steps: int = -1,
    ):
        return (
            self.predict_batch(
                queries=queries,
                answers=answers,
                documents=documents,
                generate_from=generate_from,
                num_inference_steps=num_inference_steps,
            ),
            "output1",
        )
