import os
import logging
import warnings

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.CRITICAL)
logging.getLogger("ragas").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import asyncio
import numpy as np
from typing import Any

from openai import OpenAI, AsyncOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbeddings

from app.services.rag_service import fetch_rag_context
from app.services.llm_service import sync_chat
from app.models import ChatMessage, MessageRole

class NativeOpenAIEmbeddings(BaseRagasEmbeddings):
    """
    Drop-in ragas embedding provider backed by the native OpenAI SDK.

    Implements the four methods that ragas metrics actually call:
        embed_query        ← AnswerRelevancy uses this for sync scoring
        embed_documents    ← used for batch document embedding
        aembed_query       ← async variant called by ragas' internal executor
        aembed_documents   ← async batch variant
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._sync_client = OpenAI()
        self._async_client = AsyncOpenAI()
        self._model = model

    # ── sync interface ────────────────────────────────────────────────────────
    def embed_query(self, text: str) -> list[float]:
        response = self._sync_client.embeddings.create(
            input=[text], model=self._model
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._sync_client.embeddings.create(
            input=texts, model=self._model
        )
        return [item.embedding for item in response.data]

    # ── async interface ───────────────────────────────────────────────────────
    async def aembed_query(self, text: str) -> list[float]:
        response = await self._async_client.embeddings.create(
            input=[text], model=self._model
        )
        return response.data[0].embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._async_client.embeddings.create(
            input=texts, model=self._model
        )
        return [item.embedding for item in response.data]


# ─────────────────────────────────────────────────────────────────────────────
# Golden dataset
# ─────────────────────────────────────────────────────────────────────────────
QUESTIONS = [
    "What are the two primary data structures introduced by Pandas?",
    "How does broadcasting work in NumPy?",
    "What does the 'I' stand for in an ARIMA model?",
]

REFERENCES = [
    "Pandas provides two primary data structures: Series (1-dimensional) and DataFrames (2-dimensional).",
    "Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding the smaller array to match the shape of the larger one.",
    "The 'I' stands for Integrated, which uses differencing to make the time series stationary.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Build the evaluation dataset by running your real RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────
async def generate_eval_data() -> EvaluationDataset:
    data_list = []

    for idx, (q, ref) in enumerate(zip(QUESTIONS, REFERENCES), start=1):
        print(f"  [{idx}/{len(QUESTIONS)}] {q[:70]}...")

        context_str = await fetch_rag_context(
            query=q,
            db_provider="pinecone",
            index_name="test-hybrid-collection",
            dimension=1536,
            top_k=10,
            rerank_top_n=5,
        )

        # Split the concatenated chunks; guard against blank entries
        contexts_list: list[str] = (
            [c.strip() for c in context_str.split("\n\n---\n\n") if c.strip()]
            if context_str
            else ["No context retrieved."]
        )

        system_prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer "
            "the question. If the context does not contain the answer, say so.\n\n"
            f"CONTEXT:\n{context_str}"
        )
        messages = [ChatMessage(role=MessageRole.user, content=q)]
        answer, _ = await sync_chat(messages=messages, system_prompt=system_prompt)

        data_list.append(
            {
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts_list,
                "reference": ref,
            }
        )

    return EvaluationDataset.from_list(data_list)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_evaluation() -> None:
    print("\nGenerating responses from RAG pipeline...")
    eval_dataset = await generate_eval_data()

    print("\nInitialising evaluators...")

    # LLM provider — llm_factory is the non-deprecated ragas 0.4.x API
    openai_client = OpenAI()
    ragas_llm = llm_factory("gpt-4o-mini", client=openai_client)

    # Embedding provider — our custom class; no langchain, no deprecation warnings
    ragas_embeddings = NativeOpenAIEmbeddings(model="text-embedding-3-small")

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print("Running RAGAS evaluation...\n")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # ── Display results ───────────────────────────────────────────────────────
    # result.scores is a List[Dict[str, float]] (one dict per sample).
    # We compute the aggregate (mean) ourselves — never call .items() on it.
    try:
        df = result.to_pandas()
        metric_cols = [
            c for c in df.columns
            if c not in {"user_input", "response", "retrieved_contexts", "reference"}
        ]
        aggregate: dict[str, float] = {
            col: float(np.nanmean(df[col].values)) for col in metric_cols
        }
    except Exception:
        # Absolute fallback: parse str(result) which ragas always formats as a dict
        import ast
        aggregate = ast.literal_eval(str(result))

    width = max(len(k) for k in aggregate)
    print("=" * 52)
    print("  RAGAS Evaluation Results")
    print("=" * 52)
    for metric, score in aggregate.items():
        bar = "X" * int((score if not np.isnan(score) else 0) * 20)
        flag = "  WARNING: nan -- check embeddings" if np.isnan(score) else ""
        print(f"  {metric:<{width}} : {score:.4f}  {bar}{flag}")
    print("=" * 52)

    # Per-sample breakdown + CSV export
    try:
        df = result.to_pandas()
        print("\nPer-sample scores:")
        display_cols = metric_cols + ["user_input"]
        print(df[display_cols].to_string(index=False))

        csv_path = "ragas_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nFull results saved -> {csv_path}")
    except Exception as exc:
        print(f"\n(DataFrame export skipped: {exc})")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("-" * 52)
    print("  RAGAS Evaluation")
    print("-" * 52)
    asyncio.run(run_evaluation())
    print("\nDone.")