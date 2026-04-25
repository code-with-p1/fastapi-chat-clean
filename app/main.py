from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
from pydantic import BaseModel, model_validator, Field
from typing import List
from app.models import SetupIndexRequest, IngestRequest, QueryRequest, DirectoryIngestRequest, ParentChildIngestRequest, ParentChildDirectoryIngestRequest
from app.vector_factory import get_vector_store
from app.services.chunking_factory import get_chunker, extract_and_chunk_directory
from app.vectordb.reranker import rerank_results
from app.services.parent_child_service import ParentChildProcessor

from app.config import get_settings
from app.middleware.tracing import TracingMiddleware
from app.routes import chat, health
from app.services import redis_service

settings = get_settings()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══ %s v%s starting ═══", settings.app_name, settings.app_version)
    await redis_service.init_redis()
    yield
    await redis_service.close_redis()
    logger.info("Shutdown complete ✓")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Async chat API — FastAPI + LangChain + OpenAI streaming + Redis sessions",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-Request-ID", "X-Session-ID"],
    )
    app.add_middleware(TracingMiddleware)

    app.include_router(health.router)
    app.include_router(chat.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "type": type(exc).__name__})

    return app

app = create_app()

@app.post("/api/setup_index")
async def setup_database_index(req: SetupIndexRequest):
    """
    Dynamically creates the schema, collections, and index configurations 
    tailored to the specific vector database provider.
    """
    try:
        store = get_vector_store(req.db_provider)
        result_message = store.create_index(
            index_name=req.index_name, 
            dimension=req.dimension, 
            recreate=req.recreate
        )
        return {
            "status": "success", 
            "provider": req.db_provider,
            "message": result_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_documents(req: IngestRequest):
    try:
        store = get_vector_store(req.db_provider)
        store.set_index(req.index_name)
        store.ingest(req.documents, req.dimension)
        return {"status": "success", "message": f"Ingested {len(req.documents)} docs into {req.db_provider}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve")
async def retrieve_and_rerank(req: QueryRequest):
    try:
        # 1. Dynamic DB Selection & Hybrid Search
        store = get_vector_store(req.db_provider)
        store.set_index(req.index_name)
        initial_results = store.hybrid_search(req.query, req.dimension, top_k=req.top_k)
        
        # 2. Automatic Cohere Re-Ranking
        final_results = rerank_results(
            query=req.query, 
            retrieved_docs=initial_results, 
            top_n=req.rerank_top_n
        )
        
        # 3. Formulate RAG Context
        context = "\n".join([doc["text"] for doc in final_results])
        
        return {
            "query": req.query,
            "provider": req.db_provider,
            "context_ready": context,
            "metadata": final_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/directory")
async def ingest_local_directory(req: DirectoryIngestRequest):
    """
    Scans a local directory for PDFs, chunks them according to the specified strategy, 
    and ingests them into the dynamic vector database.
    """
    try:
        # 1. Validate Directory
        if not os.path.isdir(req.directory_path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory_path}")

        # 2. Instantiate Dynamic Chunker
        chunker = get_chunker(
            strategy=req.chunking_strategy,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            semantic_threshold_type=req.semantic_threshold_type
        )

        # 3. Read and Chunk PDFs
        chunks = extract_and_chunk_directory(req.directory_path, chunker)

        if not chunks:
            return {"status": "warning", "message": "No valid text could be extracted."}

        # 4. Route to Vector DB
        store = get_vector_store(req.db_provider)
        store.set_index(req.index_name)
        store.ingest(corpus=chunks, dimension=req.dimension)

        return {
            "status": "success", 
            "message": f"Successfully extracted, chunked, and ingested {len(chunks)} chunks into {req.db_provider}."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/ingest/parent-child")
async def ingest_parent_child(req: ParentChildIngestRequest):
    try:
        processor = ParentChildProcessor(req.db_provider, req.index_name, req.dimension)
        parents, children = await processor.ingest_document(req.text)
        return {"status": "success", "message": f"Ingested {parents} parents and {children} children."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/retrieve/parent-child")
async def retrieve_parent_child(req: QueryRequest):
    try:
        processor = ParentChildProcessor(req.db_provider, req.index_name, req.dimension)
        parents = await processor.retrieve_parents(req.query, top_k=req.top_k)
        return {"query": req.query, "context_ready": "\n\n---\n\n".join(parents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/parent-child/directory")
async def ingest_parent_child_directory(req: ParentChildDirectoryIngestRequest):
    """
    Scans a local directory for PDFs, creates Parent-Child chunks, 
    stores Parents in Redis and Children in the Vector DB.
    """
    try:
        processor = ParentChildProcessor(req.db_provider, req.index_name, req.dimension)
        parents, children = await processor.ingest_directory(req.directory_path)
        
        if parents == 0 and children == 0:
            return {"status": "warning", "message": "No valid text could be extracted or no PDFs found."}
            
        return {
            "status": "success", 
            "message": f"Successfully ingested {parents} parents and {children} children from {req.directory_path} into {req.db_provider}."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
