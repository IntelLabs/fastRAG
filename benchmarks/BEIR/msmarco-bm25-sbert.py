import logging
import os
import pathlib

import tqdm
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, SentenceTransformersRanker

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

document_store = ElasticsearchDocumentStore(host="localhost", index="msmarco_index", port=80)

retriever = BM25Retriever(document_store=document_store, top_k=100)

reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])


def retrieve_queries(queries: dict, k: int):
    ans = {}
    for q, query in tqdm.tqdm(queries.items(), "queries"):
        ans[q] = {
            doc.id: doc.score
            for doc in p.run(query, params={"Reranker": {"top_k": k}})["documents"]
        }
    return ans


logging.info("Querying documents...")
results = retrieve_queries(queries, max(k_values))

#### Evaluate the retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(beir_retriever.k_values))
ndcg, _map, recall, precision = beir_retriever.evaluate(qrels, results, beir_retriever.k_values)
