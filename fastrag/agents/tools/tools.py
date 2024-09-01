import json
from typing import Dict, List, Optional, Union

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from tqdm import tqdm

from fastrag.agents.base import Tool
from fastrag.agents.utils import Color, load_text

COMPONENT_WITH_STORE = "retriever"


class HaystackPipelineContainer:
    def load_pipeline(self, pipeline_or_yaml_file):
        if isinstance(pipeline_or_yaml_file, Pipeline):
            return pipeline_or_yaml_file

        return Pipeline.loads(load_text(pipeline_or_yaml_file))


class HaystackQueryTool(Tool, HaystackPipelineContainer):
    def __init__(
        self,
        name: str,
        description: str = "",
        logging_color: Color = Color.YELLOW,
        pipeline_or_yaml_file=None,
        component_with_store=COMPONENT_WITH_STORE,
    ):
        super().__init__(
            name=name,
            description=description,
            logging_color=logging_color,
        )
        self.pipeline = self.load_pipeline(pipeline_or_yaml_file)
        self.component_with_store = component_with_store

    def query_function(self, query, params):
        """
        Create the default querying of the pipeline, by inserting the input query into all mandatory fields.
        """
        pipe_inputs = self.pipeline.inputs()
        query_dict = {}
        for inp_node_name, inp_node_params in pipe_inputs.items():
            for param_name, param_values in inp_node_params.items():
                if param_values["is_mandatory"]:
                    if inp_node_name not in query_dict:
                        query_dict[inp_node_name] = {}

                    query_dict[inp_node_name][param_name] = query

        return query_dict

    def query(self, query, params):
        run_params = self.query_function(query, params)
        return self.pipeline.run(run_params)

    def get_store(self):
        return self.pipeline.get_component(self.component_with_store).document_store

    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        result = self.query(tool_input, params)

        result_dict = result["prompt_builder"]
        return result_dict["prompt"]


class DocWithImageHaystackQueryTool(HaystackQueryTool):
    def __init__(
        self,
        name: str,
        description: str = "",
        logging_color: Color = Color.YELLOW,
        pipeline_or_yaml_file=None,
        component_with_store=COMPONENT_WITH_STORE,
    ):
        super().__init__(
            name=name,
            description=description,
            logging_color=logging_color,
            pipeline_or_yaml_file=pipeline_or_yaml_file,
            component_with_store=component_with_store,
        )

    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        result = self.query(tool_input, params)

        result_dict = result["prompt_builder"]
        tool_result = result_dict["prompt"]

        del result_dict["prompt"]
        additional_params = result_dict
        return tool_result, additional_params


class HaystackIndexTool(Tool, HaystackPipelineContainer):
    def __init__(
        self,
        name: str,
        description: str = "",
        logging_color: Color = Color.YELLOW,
        pipeline_or_yaml_file=None,
        document_store=None,
    ):
        super().__init__(
            name=name,
            description=description,
            logging_color=logging_color,
        )
        self.pipeline = self.load_pipeline(pipeline_or_yaml_file)
        self.document_store = document_store
        self.add_document_writer_to_pipeline()

    def get_index_params(self, docs):
        """
        Create the default indexing params of the pipeline, by inserting the docs into all mandatory fields.
        """
        pipe_inputs = self.pipeline.inputs()
        param_dict = {}
        for inp_node_name, inp_node_params in pipe_inputs.items():
            for param_name, param_values in inp_node_params.items():
                if param_values["is_mandatory"]:
                    if inp_node_name not in param_dict:
                        param_dict[inp_node_name] = {}

                    param_dict[inp_node_name][param_name] = docs

        return param_dict

    def index(self, docs):
        """
        Index the given documents into the pipeline.
        """
        index_params = self.get_index_params(docs)
        self.pipeline.run(index_params)

    def get_default_output(self):
        """
        Get the default output name of the pipeline.
        """
        pipeline_outputs = self.pipeline.outputs()
        component_name = list(pipeline_outputs.keys())[0]
        output_name = list(pipeline_outputs[component_name].keys())[0]
        return f"{component_name}.{output_name}"

    def add_document_writer_to_pipeline(self):
        """
        Add a DocumentWriter component to the pipeline.
        """
        pipeline_output_name = self.get_default_output()

        self.pipeline.add_component(
            instance=DocumentWriter(document_store=self.document_store), name="doc_writer"
        )

        self.pipeline.connect(pipeline_output_name, "doc_writer.documents")

    def example_to_doc(self, ex):
        return Document(content=ex["content"], meta={"title": ex["title"]})

    def run(self, tool_input: Union[str, List[dict]], params: Optional[dict] = None) -> str:
        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)

        elif isinstance(tool_input, dict) and "docs" in tool_input:
            tool_input = tool_input["docs"]

        docs = [self.example_to_doc(ex) for ex in tqdm(tool_input)]

        self.index(docs)


class DocWithImageHaystackIndexTool(HaystackIndexTool):
    def __init__(
        self,
        name: str,
        description: str = "",
        logging_color: Color = Color.YELLOW,
        pipeline_or_yaml_file=None,
        document_store=None,
    ):
        super().__init__(
            name=name,
            description=description,
            logging_color=logging_color,
            pipeline_or_yaml_file=pipeline_or_yaml_file,
            document_store=document_store,
        )

    def example_to_doc(self, ex):
        return Document(
            content=ex["content"], meta={"title": ex["title"], "image_url": ex["image_url"]}
        )


class DocWithImageFromProvidersHaystackIndexTool(DocWithImageHaystackIndexTool):
    def __init__(
        self,
        name: str,
        description: str = "",
        logging_color: Color = Color.YELLOW,
        pipeline_or_yaml_file=None,
        tool_provider_map: Dict[str, Tool] = None,
        tool_provider_name: str = None,
    ):
        # get the store from the correct tool with the document store to use
        document_store = tool_provider_map[tool_provider_name].get_store()
        super().__init__(
            name=name,
            description=description,
            logging_color=logging_color,
            pipeline_or_yaml_file=pipeline_or_yaml_file,
            document_store=document_store,
        )


TOOLS_FACTORY = {
    "doc_with_image": DocWithImageHaystackQueryTool,
    "doc_with_image_index": DocWithImageHaystackIndexTool,
    "doc_with_image_index_from_provider": DocWithImageFromProvidersHaystackIndexTool,
    "doc": HaystackQueryTool,
}
