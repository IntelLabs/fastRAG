import base64
import math
from io import BytesIO

from haystack.lazy_imports import LazyImport
from transformers import Blip2ForConditionalGeneration, Blip2Processor

with LazyImport("Make sure to have transformers >= 4.36") as llava_import:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image


def base64_to_image(base_64_input):
    bytes_io = BytesIO(base64.b64decode(base_64_input))
    return Image.open(bytes_io)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def clean_pipeline_kwargs(pipeline_kwargs):
    if "model_kwargs" in pipeline_kwargs:
        pipeline_kwargs.update(pipeline_kwargs["model_kwargs"])

    for k in [
        "task",
        "model",
        "tokenizer",
        "feature_extractor",
        "device",
        "model_kwargs",
        "pipeline_class",
        "use_fast",
        "timeout",
    ]:
        if k in pipeline_kwargs:
            del pipeline_kwargs[k]
    return pipeline_kwargs


class BaseVQAManager:
    def __init__(self):
        pillow_import.check()

    def get_pipeline(self):
        # return the model itself as a pipeline
        return self.model


class LlavaManager(BaseVQAManager):
    def __init__(
        self,
        pretrained_model_name_or_path="llava-hf/llava-1.5-7b-hf",
        pipeline_kwargs={},
    ):
        super().__init__()
        llava_import.check()
        pipeline_kwargs = clean_pipeline_kwargs(pipeline_kwargs)
        self.pipeline_kwargs = pipeline_kwargs

        self.model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **self.pipeline_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model.tokenizer = self.processor.tokenizer

    def clean_model_input_kwargs(self, model_input_kwargs):
        return {k: v for k, v in model_input_kwargs.items() if k not in ["return_full_text"]}

    def generate(
        self,
        query,
        image_obj,
        model_input_kwargs={},
    ):
        raw_image = base64_to_image(image_obj)
        query_parts = query.split("USER: ")
        prompt_prefix, after_last_user = "USER: ".join(query_parts[:-1]), query_parts[-1]

        prompt = f"{prompt_prefix}USER: <image>\n{after_last_user}"

        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(
            self.model.device
        )

        # Generate
        model_input_kwargs = self.clean_model_input_kwargs(model_input_kwargs)
        generate_ids = self.model.generate(**inputs, **model_input_kwargs)

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        outputs = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return outputs


class BlipManager(BaseVQAManager):
    def __init__(
        self,
        pretrained_model_name_or_path,
        pipeline_kwargs={},
    ):
        super().__init__()
        pipeline_kwargs = clean_pipeline_kwargs(pipeline_kwargs)
        processor = Blip2Processor.from_pretrained(pretrained_model_name_or_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, **pipeline_kwargs
        )

        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        self.model = model
        self.model.tokenizer = processor.tokenizer
        self.processor = processor

    def generate(
        self,
        query,
        image_obj,
        model_input_kwargs={},
    ):
        raw_image = base64_to_image(image_obj).convert("RGB")
        inputs = self.processor(raw_image, query, return_tensors="pt").to(self.model.device)

        out = self.model.generate(**inputs, do_sample=False)
        outputs = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return outputs


MAP_VQA_MANAGER = {
    "LlavaForConditionalGeneration": LlavaManager,
    "Blip2ForConditionalGeneration": BlipManager,
}


def get_vqa_manager(config):
    return MAP_VQA_MANAGER[config.architectures[0]]
