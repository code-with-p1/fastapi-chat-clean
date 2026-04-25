import asyncio
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.vector_factory import get_vector_store 
from app.vectordb.reranker import rerank_results
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

async def generate_hyde_query(short_query: str) -> str:
    """Generates a hypothetical answer to improve semantic retrieval."""
    logger.info("Query is short. Dynamically applying HyDE...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    messages = [
        SystemMessage(content="You are an expert financial and data science AI. Write a brief, factual, hypothetical paragraph answering the user's question to assist with semantic vector search. Do not include introductory filler."),
        HumanMessage(content=short_query)
    ]
    response = await llm.ainvoke(messages)
    # We combine the original query with the hypothetical answer
    return f"{short_query}\n{response.content}"

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
        # DYNAMIC HyDE LOGIC
        search_query = query
        word_count = len(query.split())
        if word_count < 7:
            search_query = await generate_hyde_query(query)

        def _run_pipeline():
            store = get_vector_store(db_provider)
            store.set_index(index_name)
            # Use the HyDE enhanced query for the actual retrieval
            initial_results = store.hybrid_search(search_query, dimension, top_k=top_k)
            # Use the ORIGINAL query for cross-encoder reranking to maintain strict intent
            return rerank_results(query=query, retrieved_docs=initial_results, top_n=rerank_top_n)

        final_results = await asyncio.to_thread(_run_pipeline)
        
        if not final_results:
            return None
            
        context_chunks = [doc["text"] for doc in final_results]
        return "\n".join(context_chunks)
        
    except Exception as exc:
        logger.error(f"RAG retrieval pipeline failed for {db_provider}: {exc}")
        return None