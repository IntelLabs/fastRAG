import torch
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.pipelines import Pipeline
from haystack.pipelines.base import read_pipeline_config_from_yaml

### Essential imports for loading components and modifications to original Haystack components
from fastrag import kg_creators, prompters, rankers, readers, retrievers, stores
from fastrag.utils import add_timing_to_pipeline

__version__ = "2.0.0rc0"


def load_pipeline(path):
    """Initialize a pipeline from a given yaml file path.
    Loads the invocation layers and types as classes inside the yaml config.

    returns: Pipeline
    """
    config = read_pipeline_config_from_yaml(path)
    pipe = Pipeline.load_from_config(config)
    return pipe


### Patch Pipeline to add timing measurements
Pipeline.run = add_timing_to_pipeline(Pipeline.run)
Pipeline.run_batch = add_timing_to_pipeline(Pipeline.run_batch)
