import argparse
import logging
import os
from pathlib import Path

from fastrag.utils import init_cls, init_haystack_cls, load_yaml

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create an index using FAISS as a backend")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--index-save-path", type=Path, required=True)

    args = parser.parse_args()

    Path(args.index_save_path).mkdir(parents=True, exist_ok=True)

    store_params = load_yaml(args.store)
    data_params = load_yaml(args.data)
    emb_params = load_yaml(args.embedder)

    store_cls = store_params.pop("type")
    store_params["faiss_index_path"] = None
    store_params["faiss_config_path"] = None
    store_params["sql_url"] = f"sqlite:///{args.index_save_path}" + os.sep + "faiss_doc_store.db"
    store = init_haystack_cls(store_cls, store_params)

    logger.info("Loaded store backend")
    data_cls = data_params["type"]
    data_params.pop("type")
    data = init_cls(data_cls, data_params)
    logger.info("Done loading dataset")

    logger.info("Indexing documents")
    data.chunks = 3
    for docs in data:
        store.write_documents(docs, batch_size=len(docs))
    logger.info("Done.")

    logger.info("Loading Embedder")
    emb_cls = emb_params.pop("type")
    emb_params["document_store"] = store
    emb = init_haystack_cls(emb_cls, emb_params)
    logger.info("Encoding vectors")
    store.update_embeddings(emb, batch_size=emb_params["batch_size"])
    index_path = args.index_save_path / "faiss_vec"
    index_config_path = args.index_save_path / "faiss_vec.json"
    logger.info("Saving index")
    store.save(str(index_path), str(index_config_path))
    logger.info("Done.")
