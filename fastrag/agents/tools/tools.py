from typing import Optional

from haystack import Document
from tqdm import tqdm

from fastrag.agents.base import Tool


class DocWithImageTool(Tool):
    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        result = self.query_handler.query(tool_input, params)

        result_dict = result["prompt_builder"]
        tool_result = result_dict["prompt"]
        
        del result_dict["prompt"]
        additional_params = result_dict
        return tool_result, additional_params

    def upload_data_to_pipeline(self, params):
        """
        Upload documents into the textual with images document store.
        """
        # ensure that params contain the text to upload
        if "data_rows" not in params:
            return

        for data_rows in params["data_rows"]:
            docs = [
                Document(
                    content=ex["content"], meta={"title": ex["title"], "image_url": ex["image_url"]}
                )
                for ex in tqdm(data_rows)
            ]
            self.index_handler.index(docs)


class DocTool(Tool):
    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        result = self.query_handler.query(tool_input, params)

        result_dict = result["prompt_builder"]
        text = result_dict["prompt"]

        tool_result = text

        return tool_result

    def upload_data_to_pipeline(self, params):
        """
        Upload documents into the textual with images document store.
        """
        # ensure that params contain the text to upload
        if "data_rows" not in params:
            return

        for data_rows in params["data_rows"]:
            docs = [
                Document(content=ex["content"], meta={"title": ex["title"]})
                for ex in tqdm(data_rows)
            ]
            self.index_handler.index(docs)


TOOLS_FACTORY = {"doc_with_image": DocWithImageTool, "doc": DocTool}
