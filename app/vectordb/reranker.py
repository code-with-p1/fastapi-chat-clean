import os
import cohere
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

co = cohere.Client(os.environ.get("COHERE_API_KEY"))

def rerank_results(query: str, retrieved_docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    if not retrieved_docs:
        return []
    
    docs_text = [doc["text"] for doc in retrieved_docs]
    
    response = co.rerank(
        query=query,
        documents=docs_text,
        top_n=top_n,
        model='rerank-english-v3.0'
    )
    
    # Map back original document data with new relevance scores
    reranked = []
    for res in response.results:
        original_doc = retrieved_docs[res.index]
        reranked.append({
            "text": original_doc["text"],
            "original_score": original_doc["score"],
            "rerank_score": res.relevance_score
        })
        
    return reranked