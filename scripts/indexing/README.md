# Indexing with fastRAG

fastRAG can be used with any Haystack-based indexing store (which levrages on Haystack's `DocumentStore` class).
fastRAG includes a directory `scripts/indexing/` with scripts for creating indexes for all of fastRAG supported pipelines.

**1. Elasticsearch**:

For creating an Elasticsearch index (used with BM25 sparse retriever), the following script can be used:

```sh
python scripts/indexing/create_elastic.py \
    --store config/store/elastic-local.yaml \
    --data config/data/wikipedia_w100_hfdataset.yaml
```

**2. FAISS**:

For creating a FAISS-based dense index with DPR as an embedder/retriver, the following script can be used:

```sh
python scripts/indexing/create_faiss.py \
    --store config/store/faiss.yaml \
    --data config/data/wikipedia_w100_hfdataset.yaml \
    --embedder config/retriever/dpr.yaml \
    --index-save-path <save path>
```

**3. Qdrant + SentenceTransformers**:

For creating a [Qdrant](https://qdrant.tech/)-based dense index with a sentence-transformer model as an embedder/retriver, the following script can be used:

```sh
python scripts/indexing/create_dense.py                          \
       --data config/data/wikipedia_hf_6M.yaml                   \
       --embedder config/embedder/sentence-transformer-docs.yaml \
       --store config/store/qdrant.yaml
```

**4. PLAID**:

PLAID (Based on [this](https://doi.org/10.48550/arXiv.2205.09707) paper) is a dense retrieval index engine that stores token vectors using an efficient algorithm. PLAID must be used with dense token embedder such as ColBERT which can embed tokens and utlizes a token-to-token ranking similarity method for ranking documents.
More info on PLAID can be found in our [models](../../models) page.

**PLAID Requirements**:

1. Indexing with a GPU is supported with a RTX 3090 (Ampere) or newer and PyTorch should be installed with CUDA support using:

    ```sh
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```

2. PLAID utilized `faiss` for running kmeans clustering. For higher performance it is required to install `faiss-gpu` (for both CPU/GPU backends) via `conda` package manager. See [this page for detailed instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-from-conda-forge).

For creating a PLAID-based dense index, a ColBERT checkpoint is reuired in addition to a corpus and store configuration. The following script can be used to create such index:

```sh
python scripts/indexing/create_plaid.py \
    --checkpoint=<path-to-colbert-model-checkpoint> \
    --collection=<path to tsv collection> \
    --index_save_path=<index-save-path> \
    --gpus=0 \
    --ranks=1 \
    --name=plaid_test \
    --kmeans_iterations=4
```
