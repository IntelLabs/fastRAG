import base64
import logging
from io import BytesIO
from typing import List

import requests
from haystack import component
from haystack.components.builders import PromptBuilder


class MultiModalPromptBuilder(PromptBuilder):
    """
    PromptBuilder is a component that renders a prompt from a template string using Jinja2 templates.
    The template variables found in the template string are used as input types for the component and are all required.

    Usage example:
    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```
    """

    @component.output_types(prompt=str, images=List[str])
    def run(self, **kwargs):
        """
        :param kwargs:
            The variables that will be used to render the prompt template.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the prompt template.
        """
        prompt_dict = {"prompt": self.template.render(kwargs)}
        prompt_dict["images"] = [
            self.get_base64_from_url(doc.meta["image_url"]) for doc in kwargs["documents"]
        ]

        return prompt_dict

    def get_base64_from_url(self, image_url):
        response = requests.get(
            image_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            },
        )
        buffered = BytesIO(response.content)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


component._component(MultiModalPromptBuilder)
