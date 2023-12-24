import logging
from pathlib import Path
from typing import Optional, Union

from haystack.document_stores import FAISSDocumentStore

logger = logging.getLogger(__name__)


class FastRAGFAISSStore(FAISSDocumentStore):
    def __init__(
        self,
        sql_url: str = "sqlite:///faiss_document_store.db",
        vector_dim: Optional[int] = None,
        embedding_dim: int = 768,
        faiss_index_factory_str: str = "Flat",
        faiss_index=None,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "dot_product",
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        faiss_index_path: Optional[Union[str, Path]] = None,
        faiss_config_path: Optional[Union[str, Path]] = None,
        isolation_level: Optional[str] = None,
        n_links: int = 64,
        ef_search: int = 20,
        ef_construction: int = 80,
        validate_index_sync: bool = True,
    ):
        if faiss_index_path is None:
            try:
                file_path = sql_url.split("sqlite:///")[1]
                logger.error(file_path)
            except OSError:
                pass
            validate_index_sync = False
            faiss_index_path = None
            faiss_config_path = None
            super().__init__(
                sql_url=sql_url,
                validate_index_sync=validate_index_sync,
                vector_dim=vector_dim,
                embedding_dim=embedding_dim,
                faiss_index_factory_str=faiss_index_factory_str,
                faiss_index=faiss_index,
                return_embedding=return_embedding,
                index=index,
                similarity=similarity,
                embedding_field=embedding_field,
                progress_bar=progress_bar,
                duplicate_documents=duplicate_documents,
                isolation_level=isolation_level,
                n_links=n_links,
                ef_search=ef_search,
                ef_construction=ef_construction,
            )
        else:
            validate_index_sync = True
            sql_url = None
            super().__init__(
                faiss_index_path=faiss_index_path,
                faiss_config_path=faiss_config_path,
            )

        total_gpus_to_use = faiss.get_num_gpus()
        use_gpu = True if total_gpus_to_use > 0 else False
        if use_gpu:
            faiss_index_cpu = self.faiss_indexes["document"]
            total_gpus = faiss.get_num_gpus()
            if total_gpus_to_use is not None:
                total_gpus_to_use = min(total_gpus, total_gpus_to_use)
                faiss_index_gpu = faiss.index_cpu_to_gpus_list(
                    index=faiss_index_cpu, ngpu=total_gpus_to_use
                )
                logger.info(f"Faiss index uses {total_gpus_to_use} gpus out of {total_gpus}")
            else:
                faiss_index_gpu = faiss.index_cpu_to_all_gpus(faiss_index_cpu)
                logger.info(f"Faiss index uses all {total_gpus} gpus")

            if faiss_index_path is not None:
                assert faiss_index_gpu.ntotal > 0
            logger.info(f"Faiss gpu index size: {faiss_index_gpu.ntotal}")
            super().__init__(sql_url=sql_url, faiss_index=faiss_index_gpu)
