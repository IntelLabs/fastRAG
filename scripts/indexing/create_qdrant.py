import argparse
import logging
from pathlib import Path

from fastrag.utils import init_cls, init_haystack_cls, load_yaml

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create an index using Qdrant as a backend")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=False)

    args = parser.parse_args()

    store_params = load_yaml(args.store)
    data_params = load_yaml(args.data)
    emb_params = load_yaml(args.embedder)

    store_cls = store_params.pop("type")
    store = init_haystack_cls(store_cls, store_params)
    logger.info("Loaded store backend")

    data_cls = data_params.pop("type")
    data = init_cls(data_cls, data_params)
    logger.info("Done loading dataset")

    logger.info("Indexing documents")
    for docs in data:
        store.write_documents(docs, batch_size=args.batch_size or 100)
    logger.info("Done.")

    logger.info("Loading Embedder")
    emb_cls = emb_params.pop("type")
    emb_params["document_store"] = store
    emb = init_haystack_cls(emb_cls, emb_params)

    logger.info("Encoding vectors")
    store.update_embeddings(emb, batch_size=emb_params["batch_size"])
    logger.info("Done.")
