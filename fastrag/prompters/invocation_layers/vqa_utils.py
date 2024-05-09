import base64
import logging
import math
from io import BytesIO

from haystack.lazy_imports import LazyImport
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor

with LazyImport("Make sure to have transformers >= 4.36") as llava_import:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image

with LazyImport("Run 'pip install llama-cpp-python'") as llava_llama_cpp_import:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler


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

    def clear_cached(self):
        if hasattr(self, "image_str"):
            del self.image_str


class LlavaLlamaCPPManager(BaseVQAManager):
    def __init__(
        self,
        pretrained_model_name_or_path,
        pipeline_kwargs={},
    ):
        super().__init__()
        llava_llama_cpp_import.check()

        pipeline_kwargs = clean_pipeline_kwargs(pipeline_kwargs)

        if "clip_model_path" not in pipeline_kwargs:
            raise Exception(
                "No clip_model_path provided. Please add a 'clip_model_path' to the pipeline_kwargs."
            )

        if "tokenizer_name_or_path" not in pipeline_kwargs:
            raise Exception(
                "No tokenizer_name_or_path provided. Please add a 'tokenizer_name_or_path' to the pipeline_kwargs."
            )

        clip_model_path = pipeline_kwargs["clip_model_path"]

        chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path)

        self.model_max_length = pipeline_kwargs.get("model_max_length", 4096)
        self.n_threads = pipeline_kwargs.get("n_threads", None)
        self.numa = pipeline_kwargs.get("numa", False)
        self.stream = pipeline_kwargs.get("numa", False)

        self.model = Llama(
            model_path=pretrained_model_name_or_path,
            chat_handler=chat_handler,
            n_ctx=self.model_max_length,  # n_ctx should be increased to accomodate the image embedding
            logits_all=True,  # needed to make llava work
            verbose=False,
            n_threads=self.n_threads,
            numa=self.numa,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pipeline_kwargs["tokenizer_name_or_path"])
        self.device = pipeline_kwargs.get("device", "cpu")

    def get_pipeline(self):
        # return the model itself as a pipeline
        return self

    def generate(
        self,
        query,
        image_obj,
        model_input_kwargs={},
    ):
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
        ]
        user_message = {"role": "user", "content": []}
        if image_obj is None and hasattr(self, "image_str"):
            image_obj = self.image_str

        if image_obj is not None:
            user_message["content"].append(
                {"type": "image_url", "image_url": {"url": "data:," + image_obj}}
            )

        user_message["content"].append({"type": "text", "text": query})
        messages.append(user_message)
        stream = "streamer" in model_input_kwargs
        output = self.model.create_chat_completion(messages, stream=stream)

        total_text = ""
        if stream:
            for gen_token in output:
                delta = gen_token["choices"][0]["delta"]
                current_token = delta["content"] if delta is not None and "content" in delta else ""
                model_input_kwargs["streamer"].token_handler(current_token)
                total_text += current_token
        else:
            total_text = output["choices"][0]["message"]["content"]
        return total_text


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
        if image_obj is None and hasattr(self, "image_str"):
            image_obj = self.image_str

        if image_obj is not None:
            raw_image = base64_to_image(image_obj)

            prompt = query

            if "<image>" not in prompt:
                prompt = "Current Image: <image>\n" + prompt

            inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(
                self.model.device
            )
        else:
            inputs = self.processor(text=query, return_tensors="pt")

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
    "llava_llama_cpp": LlavaLlamaCPPManager,
}


def get_vqa_manager(vqa_type):
    return MAP_VQA_MANAGER[vqa_type]
