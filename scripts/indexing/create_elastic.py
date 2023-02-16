import argparse
import logging
from pathlib import Path

from fastrag.utils import init_cls, init_haystack_cls, load_yaml

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create an index using Elasticsearch as a backend")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)

    args = parser.parse_args()

    store_params = load_yaml(args.store)
    data_params = load_yaml(args.data)

    store_cls = store_params.pop("type")
    store = init_haystack_cls(store_cls, store_params)
    logger.info("Loaded store backend")

    data_cls = data_params.pop("type")
    data = init_cls(data_cls, data_params)
    logger.info("Done loading dataset")

    logger.info("Indexing documents")
    for docs in data:
        store.write_documents(docs, batch_size=len(docs))
    logger.info("Done.")
