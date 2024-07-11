import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

from haystack import component
from haystack.components.generators import HuggingFaceLocalGenerator, hugging_face_local
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret
from transformers import AutoProcessor, AutoTokenizer, TextStreamer

# support for llava models
hugging_face_local.SUPPORTED_TASKS.append("image-to-text")

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image


def base64_to_image(base_64_input):
    bytes_io = BytesIO(base64.b64decode(base_64_input))
    return Image.open(bytes_io)


class LlavaHFGenerator(HuggingFaceLocalGenerator):
    """
    Generator based on a Llava Hugging Face model loaded.
    The original Llava model: https://github.com/haotian-liu/LLaVA

    This component provides an interface to generate text using a Llava Hugging Face model.

    Usage example:
    ```python

    from fastrag.generators.llava import LlavaHFGenerator

    generator = LlavaHFGenerator(
        model = "llava-hf/llava-1.5-7b-hf",
        task = "image-to-text",
        generation_kwargs = {
            "max_new_tokens": 100,
        },
    )

    print(generator.run("Who is the best American actor?"))
    # {'replies': ['John Cusack']}
    ```
    """

    def __init__(
        self,
        model: str = "llava-hf/llava-1.5-7b-hf",
        task: Optional[Literal["image-to-text"]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
    ):
        """
        Creates an instance of a LlavaHFGenerator.

        :param model: The name or path of a Hugging Face model for text generation,
        :param compressed_model_dir: The name or path of the compressed model,
        :param task: The task for the Hugging Face pipeline.
            Possible values are "text-generation" and "text2text-generation".
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If the task is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
            If not specified, the component will attempt to infer the task from the model name,
            calling the Hugging Face Hub API.
        :param device_openvino: The Intel device on which the model will be loaded, run 'openvino.Core()' to view the available devices. Defaults to CPU.
        :param ov_config: The OpenVINO configuration dictionary.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If the token is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information:
            - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
            - [transformers.GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            See Hugging Face's [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task)
            for more information on the available kwargs.
            In this dictionary, you can also include `model_kwargs` to specify the kwargs for model initialization:
            [transformers.PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
        :param stop_words: A list of stop words. If any one of the stop words is generated, the generation is stopped.
            If you provide this parameter, you should not specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, it's important to make sure your prompt has no stop words.
        """
        pillow_import.check()
        super().__init__(
            model=model,
            task=task,
            device=device,
            token=token,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs,
            stop_words=stop_words,
        )
        if "streamer" not in self.generation_kwargs:
            self.generation_kwargs["streamer"] = TextStreamer(AutoTokenizer.from_pretrained(model))

        if "stopping_criteria" in self.generation_kwargs:
            self.stopping_criteria_list = self.generation_kwargs["stopping_criteria"]
            del self.generation_kwargs["stopping_criteria"]

        self.processor = AutoProcessor.from_pretrained(model)
        self.image_token = "<image>"

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        images: list[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run the text generation model on the given prompt.

        :param prompt:
            A string representing the prompt.
        :param images:
            A list of base64 strings representing the input images.
        :param generation_kwargs:
            Additional keyword arguments for text generation.

        :returns:
            A dictionary containing the generated replies.
            - replies: A list of strings representing the generated replies.
            - raw_images: A list of the raw PIL images.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "The generation model has not been loaded. Please call warm_up() before running."
            )

        if not prompt:
            return {"replies": []}

        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        raw_images = None
        if images:
            raw_images = [base64_to_image(img) for img in images]

            present_image_token_count = prompt.count(self.image_token)
            image_token_count_diff = len(images) - present_image_token_count

            # check if we need to add additional image tokens
            if image_token_count_diff > 0:
                image_token_full_str = " ".join(
                    [self.image_token for _ in range(image_token_count_diff)]
                )
                prompt = f"Current Images: {image_token_full_str}\n" + prompt
                # prompt = f"{prompt}\n{image_token_full_str}\n"
            inputs = self.processor(prompt, raw_images, return_tensors="pt")
        else:
            inputs = self.processor(prompt, return_tensors="pt")

        updated_generation_kwargs["max_length"] = updated_generation_kwargs.get("max_length", 32000)

        output = self.pipeline.model.generate(
            **inputs, stopping_criteria=self.stopping_criteria_list, **updated_generation_kwargs
        )

        replies = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        gen_stop_words = self.get_stop_words_from_kwargs()

        stop_words = None
        stop_words = gen_stop_words if gen_stop_words else self.stop_words
        if stop_words:
            # the output of the pipeline includes the stop word
            replies = [
                reply.replace(stop_word, "").rstrip()
                for reply in replies
                for stop_word in stop_words
            ]

        return {"replies": replies, "raw_images": raw_images}

    def get_stop_words_from_kwargs(self):
        if "stopping_criteria" in self.generation_kwargs:
            for stop_crt in self.generation_kwargs["stopping_criteria"]:
                if hasattr(stop_crt, "stop_words_text"):
                    return stop_crt.stop_words_text
        return None

    def get_user_text(self, chat_snippet):
        user_text = ""
        if "additional_params" in chat_snippet and "images" in chat_snippet["additional_params"]:
            images_tokens = " ".join(
                [self.image_token for _ in range(len(chat_snippet["additional_params"]["images"]))]
            )
            user_text = f"\n\n{images_tokens}\n"
        user_text += chat_snippet["Human"]
        return user_text
