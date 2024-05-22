import argparse
import logging
from pathlib import Path

from haystack.lazy_imports import LazyImport

from fastrag.stores import PLAIDDocumentStore

with LazyImport(
    "Install Faiss using 'pip install faiss-cpu' or 'pip install faiss-gpu'"
) as faiss_import:
    import faiss

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create an index using PLAID engine as a backend")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--index-save-path", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--doc-max-length", type=int, default=120)
    parser.add_argument("--query-max-length", type=int, default=60)
    parser.add_argument("--kmeans-iterations", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=2)

    args = parser.parse_args()
    faiss_import.check()

    if args.gpus > 1:
        args.ranks = args.gpus
        args.amp = True
    assert args.ranks > 0
    if args.gpus == 0:
        assert args.ranks > 0

    store = PLAIDDocumentStore(
        index_path=f"{args.index_save_path}",
        checkpoint_path=f"{args.checkpoint}",
        collection_path=f"{args.collection}",
        create=True,
        nbits=args.nbits,
        gpus=args.gpus,
        ranks=args.ranks,
        doc_maxlen=args.doc_max_length,
        query_maxlen=args.query_max_length,
        kmeans_niters=args.kmeans_iterations,
    )
    logger.info("Done.")
