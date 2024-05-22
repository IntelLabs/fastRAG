import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from fastrag.utils import init_cls, load_yaml

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Embed data and save to pickled file.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--batch_num", type=int, required=False)

    args = parser.parse_args()

    store_params = load_yaml(args.store)
    store_cls = store_params.pop("type")
    store = init_cls(store_cls, store_params["init_parameters"])
    logger.info("Loaded store backend")

    data_params = load_yaml(args.data)
    emb_params = load_yaml(args.embedder)

    data_cls = data_params.pop("type")
    data = init_cls(data_cls, data_params)
    logger.info("Done loading dataset")

    logger.info("Loading Embedder")
    emb_cls = emb_params.pop("type")
    emb = init_cls(emb_cls, emb_params["init_parameters"])
    emb.warm_up()

    # Can start at a given batch, if provided
    batch_start = args.batch_num or 0

    logger.info("Creating Embeddings...")

    for batch_i in tqdm(
        list(range(data.chunks)),
        desc="Data chunks",
    ):
        if batch_i >= batch_start:
            batch = []
            docs = data.process(batch_i)  # requires the HFDatasetLoader
            emb_batch = emb.run(documents=docs)
            store.write_documents(emb_batch["documents"], policy="overwrite")

    logger.info("Done.")
