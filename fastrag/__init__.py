from haystack.pipelines import Pipeline

### Essential imports for loading components and modifications to original Haystack components
from fastrag import image_generators, kg_creators, rankers, readers, retrievers, stores
from fastrag.utils import add_timing_to_pipeline

__version__ = "1.2.0"


def load_pipeline(config_path: str) -> Pipeline:
    """Initialize a pipeline from a given yaml file path

    returns: Pipeline
    """
    return Pipeline.load_from_yaml(config_path)


### Patch Pipeline to add timing measurements
Pipeline.run = add_timing_to_pipeline(Pipeline.run)
Pipeline.run_batch = add_timing_to_pipeline(Pipeline.run_batch)
