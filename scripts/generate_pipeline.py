"""Generate a Haystack pipeline file given a collection of configurations"""

import argparse
import sys
from pathlib import Path

import yaml

parser = argparse.ArgumentParser("Pipeline Generator")

parser.add_argument(
    "-f", "--file", help="Specify the output file. Otherwise write to stdout.", type=Path
)
parser.add_argument("--name", help="Specify pipeline name", default="my_pipeline")
parser.add_argument("--path", help='Specify the pipeline path; e.g. "store,reader"', type=str)

# Components
parser.add_argument("--store", help="Specify the store config.", type=Path)
parser.add_argument("--retriever", help="Specify the retriever config.", type=Path)
parser.add_argument("--embedder", help="Specify the embedder config.", type=Path)
parser.add_argument("--reranker", help="Specify the reranker config.", type=Path)
parser.add_argument("--reader", help="Specify the reader config.", type=Path)
parser.add_argument("--image-gen", help="Specify the image generator config.", type=Path)

args = parser.parse_args()

steps = args.path.split(",")

components_types = ["store", "retriever", "embedder", "reranker", "reader", "image_gen"]

# Nodes Building
components = {}
for step in components_types:
    if (file_params := getattr(args, step)) is None:
        continue

    params = yaml.safe_load(file_params.open())
    node_type = params.pop("type")
    node_name = step.capitalize()
    component = {"name": node_name, "type": node_type, "params": params}
    components[node_name] = component


# (Retriever <-> Store) Matching
if args.retriever:
    if args.store is None:
        raise NotImplementedError("Can't use a retriever without a document store.")
    components["Retriever"]["params"]["document_store"] = "Store"


# Pipeline Building
previous = "Query"
nodes = []
for step in steps:
    node = {"name": step.capitalize(), "inputs": [previous]}
    nodes.append(node)
    previous = step.capitalize()


# YAML output
# Output to stdoutput
# To be used in scripts as redirection

output = yaml.dump(
    {
        "version": "1.12.2",
        "components": list(components.values()),
        "pipelines": [{"name": args.name, "nodes": nodes}],
    },
    indent=2,
)

# Writing to file or to stdout
if args.file:
    with open(args.file, "w") as w:
        w.write(output)
else:
    sys.stdout.write(output)
