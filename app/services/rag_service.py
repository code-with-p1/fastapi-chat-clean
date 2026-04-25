import asyncio
import logging
from typing import Optional
from app.vector_factory import get_vector_store 
from app.vectordb.reranker import rerank_results

logger = logging.getLogger(__name__)

async def fetch_rag_context(
    query: str, 
    db_provider: str,
    index_name: str,
    dimension : int = 1536, 
    top_k: int = 10, 
    rerank_top_n: int = 5
) -> Optional[str]:
    """
    Executes the dynamic DB hybrid search and Cohere reranking.
    Wrapped in asyncio.to_thread to prevent blocking the main event loop.
    """
    try:
        def _run_pipeline():
            store = get_vector_store(db_provider)
            store.set_index(index_name)
            initial_results = store.hybrid_search(query, dimension=dimension, top_k=top_k)
            return rerank_results(query=query, retrieved_docs=initial_results, top_n=rerank_top_n)

        final_results = await asyncio.to_thread(_run_pipeline)
        
        if not final_results:
            return None
            
        context_chunks = [doc["text"] for doc in final_results]
        return "\n".join(context_chunks)
        
    except Exception as exc:
        logger.error(f"RAG retrieval pipeline failed for {db_provider}: {exc}")
        return None