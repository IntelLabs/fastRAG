from typing import Dict, Tuple

from haystack import Pipeline
from haystack.components.writers import DocumentWriter

from fastrag.agents.utils import load_text

######## Query Handlers

COMPONENT_WITH_STORE = "retriever"


class QueryHandler:
    def query(self, query, params) -> Tuple[str, Dict]:
        observation: str = ""
        additional_params: dict = {}
        return observation, additional_params


class HaystackQueryHandler(QueryHandler):
    def __init__(self, pipeline, query_function=None, component_with_store=COMPONENT_WITH_STORE):
        self.pipeline = pipeline
        self.query_function = query_function if query_function else self.default_query_function
        self.component_with_store = component_with_store

    def default_query_function(self, query, params):
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

    def query(self, query, params) -> Tuple[str, Dict]:
        run_params = self.query_function(query, params)
        return self.pipeline.run(run_params)

    def get_store(self):
        return self.pipeline.get_component(self.component_with_store).document_store


class HaystackFileQueryHandler(HaystackQueryHandler):
    def __init__(
        self, pipeline_yaml_path, query_function=None, component_with_store=COMPONENT_WITH_STORE
    ):
        pipeline = Pipeline.loads(load_text(pipeline_yaml_path))
        super().__init__(pipeline, query_function, component_with_store)


######## Indexing Handlers


class IndexingHandler:
    def index(self, docs):
        pass


class HaystackIndexingHandler(IndexingHandler):
    def __init__(self, pipeline, document_store):
        """
        :param create_pipeline:
            A function that creates the indexing pipeline. We create it to avoid problems with stale pipelines
            in haystack v2.
        :param indexing_function:
            A function that indexes a given list of documents into the indexing pipeline.
        """
        self.pipeline = pipeline
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


class HaystackFileIndexingHandler(HaystackIndexingHandler):
    def __init__(self, pipeline_yaml_path, document_store):
        pipeline = Pipeline.loads(load_text(pipeline_yaml_path))
        super().__init__(pipeline, document_store)


QUERY_HANDLER_FACTORY = {
    "haystack": HaystackQueryHandler,
    "haystack_yaml": HaystackFileQueryHandler,
}

INDEX_HANDLER_FACTORY = {
    "haystack": HaystackIndexingHandler,
    "haystack_yaml": HaystackFileIndexingHandler,
}
