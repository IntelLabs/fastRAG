# Configurations

This folder contains pipelines and individual components configurations files needed to for creating pipelines for
inference and benchmarking.

## Full Pipelines

There are several ready-to-run full pipelines. They are compatible with the haystack pipeline API.

- `qa_pipeline` - Elasticsearch retriever, SBERT cross encoder reranker and an FiD reader.
- `qa_plaid`- ColBERT/PLAID retriever and an FiD reader.
- `summarization_pipeline` - BM25 retriever, SBERT cross encoder reranker and an instruction tuned BART-LARGE reader.

## Components Configuration

The following folders: `data`, `embedder`, `reader`, `reranker`, `retriever` and `store` contain
components' configuration YAML files. Those can be adapted to use specific models by providing a local path or a
HuggingFace hub's model name or dataset name. After components are chosen, one needs to run the [Pipeline
Generation](../scripts/generate_pipeline.py) script in order to generate a pipeline configuration which describes the
components and how they are connected. For example:

```sh
python generate_pipeline.py --path "retriever,reranker,reader"  \
       --store config/store/elastic.yaml                        \
       --retriever config/retriever/bm25.yaml                   \
       --reranker config/reranker/sbert.yaml                    \
       --file pipeline.yaml
```

This means you want to create a pipeline containing an elasticsearch based data store, a retriever based on BM25
algorithm, an SBERT cross encoding reranker and an FiD reader pipeline. The pipeline is saved as `pipeline.yaml` and can
be used to create a web server to run queries:

```sh
python -m fastrag.rest_api.application pipeline.yaml
```

See [Running Pipelines](../README.md#running-pipelines) section for more details.
