import os
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.services.rag_service import fetch_rag_context
from app.services.llm_service import sync_chat
from app.models import ChatMessage, MessageRole

# 1. The Golden Dataset (Sourced from your PDF)
# In production, this would be 50-100 questions.
QUESTIONS = [
    "What are the two primary data structures introduced by Pandas?",
    "How does broadcasting work in NumPy?",
    "What does the 'I' stand for in an ARIMA model?"
]

GROUND_TRUTHS = [
    ["Pandas provides two primary data structures: Series (1-dimensional) and DataFrames (2-dimensional)."],
    ["Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding the smaller array to match the shape of the larger one."],
    ["The 'I' stands for Integrated, which uses differencing to make the time series stationary."]
]

async def generate_eval_data():
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": GROUND_TRUTHS
    }
    
    for q in QUESTIONS:
        # Fetch Context from your existing Pinecone/Milvus pipeline
        context_str = await fetch_rag_context(
            query=q, db_provider="pinecone", top_k=5, rerank_top_n=3
        )
        contexts_list = context_str.split("\n\n---\n\n") if context_str else [""]
        
        # Generate the LLM Answer
        system_prompt = f"Use the context to answer the question. \n\nCONTEXT:\n{context_str}"
        messages = [ChatMessage(role=MessageRole.user, content=q)]
        answer, _ = await sync_chat(messages=messages, system_prompt=system_prompt)
        
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts_list)
        
    return Dataset.from_dict(data)

async def run_evaluation():
    print("Generating responses for Golden Dataset...")
    eval_dataset = await generate_eval_data()
    
    print("Running RAGAS Evaluation...")
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevance, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n--- RAGAS Evaluation Results ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(run_evaluation())