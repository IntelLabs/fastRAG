# :red_car: Getting Started

fastRAG leverages Haystack's pipelining abstraction. We recommend constructing a flow by incorporating components
provided by fastRAG and Haystack, tailored to the specific task you aim to tackle. There are various approaches to
achieving this using fastRAG.

## Defining Pipelines in your Code

To define a pipeline in your Python code, you can initialize all the components with the desired configuration directly
in your code. This allows you to have full control over the pipeline structure and parameters. For concrete examples and
detailed implementation guidance, please refer to the example [notebooks](examples/) provided by our team.

## Defining Pipelines Using YAML

Another approach to defining pipelines is by writing a YAML file following Haystack's format. This method allows for a
more declarative and modular pipeline configuration. You can find detailed information on how to define pipelines using
a YAML file in the [Haystack documentation](https://docs.haystack.deepset.ai/docs/pipelines#yaml-file-definitions). The
documentation provides guidance on the structure of the YAML file, available components, their parameters, and how to
combine them to create a custom pipeline.

We have provided miscellaneous pipeline configurations in the config directory.

## Serving a Pipeline via REST API

To serve a fastRAG pipeline through a REST API, you can follow these steps:

1. Execute the following command in your terminal:

    ```bash
    python -m fastrag.rest_api.application --config=pipeline.yaml
    ```

2. If needed, you can explore additional options using the `-h` flag.

3. The REST API service includes support for Swagger. You can access a user-friendly UI to observe and interact with the API endpoints by visiting `http://localhost:8000/docs` in your web browser.

The available endpoints for the REST API service are as follows:

- `status`: This endpoint can be used to perform a sanity check.
- `version`: This endpoint provides the project version, as defined in `__init__.py`.
- `query`: Use this endpoint to run a query through the pipeline and retrieve the results.

By leveraging the REST API service, you can integrate fastRAG pipelines into your applications and easily interact with them using HTTP requests.

## Generating Pipeline Configurations

### Generate using a script

The pipeline in fastRAG is constructed using the Haystack pipeline API and is dynamically generated based on the user's selection of components. To generate a Haystack pipeline that can be executed as a standalone REST server service (refer to [REST API](#serving-a-pipeline-via-rest-api)), you can utilize the [Pipeline Generation](scripts/generate_pipeline.py) script.

Below is an example that demonstrates how to use the script to generate a pipeline with a ColBERT retriever, and an SBERT reranker:

```bash
python generate_pipeline.py --path "retriever,reranker,reader" \
    --store config/store/plaid-wiki.yaml \
    --retriever config/retriever/colbert-v2.yaml \
    --reranker config/reranker/sbert.yaml \
    --file pipeline.yaml
```

In the above command, you specify the desired components using the `--path` option, followed by providing the corresponding configuration YAML files for each component (e.g., `--store`, `--retriever`, `--reranker`). Finally, you can specify the output file for the generated pipeline configuration using the `--file` option (in this example, it is set to `pipeline.yaml`).

## Index Creation

For detailed instructions on creating various types of indexes, please refer to the [Indexing Scripts](scripts/indexing/) directory. It contains valuable information and resources to guide you through the process of creating different types of indexes.

## Customizing Models

To cater to different use cases, we provide a variety of training scripts that allow you to fine-tune models of your choice. For detailed examples, model descriptions, and more information, please refer to the [Components Overview](components.md) page. It will provide you with valuable insights into different models and their applications.
