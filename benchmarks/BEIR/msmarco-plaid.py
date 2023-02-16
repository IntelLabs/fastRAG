import logging
import os
import pathlib

import tqdm
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from haystack import Pipeline

from fastrag.retrievers.colbert import ColBERTRetriever
from fastrag.stores import PLAIDDocumentStore

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(".").absolute(), "benchmarks", "datasets")
data_path = util.download_and_unzip(url, out_dir)

logging.info("Loading dataset...")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

k_values = [10, 20, 50, 100]

beir_retriever = EvaluateRetrieval(k_values=k_values)

logging.info("Loading PLAID index...")
document_store = PLAIDDocumentStore(
    index_path="/path/to/index",
    checkpoint_path="/path/to/checkpoint",
    collection_path="/path/to/collection.tsv",
)

retriever = ColBERTRetriever(document_store=document_store)

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])


def retrieve_queries(queries: dict, k: int):
    ans = {}
    for q, query in tqdm.tqdm(queries.items(), "queries"):
        ans[q] = {
            doc.id: doc.score
            for doc in p.run(query, params={"Retriever": {"top_k": k}})["documents"]
        }
    return ans


logging.info("Querying documents...")
results = retrieve_queries(queries, max(k_values))


#### Evaluate the retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(beir_retriever.k_values))
ndcg, _map, recall, precision = beir_retriever.evaluate(qrels, results, beir_retriever.k_values)
