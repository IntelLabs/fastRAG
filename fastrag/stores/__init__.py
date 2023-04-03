from fastrag.utils import safe_import

FaissStore = safe_import("fastrag.stores.faiss", "FaissStore")
FastRAGFAISSStore = safe_import("fastrag.stores.faiss", "FastRAGFAISSStore")
PLAIDDocumentStore = safe_import("fastrag.stores.plaid", "PLAIDDocumentStore")
QdrantDocumentStore = safe_import("qdrant_haystack", "QdrantDocumentStore")
