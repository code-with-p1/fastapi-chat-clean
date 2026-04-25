# Dynamic Hybrid RAG API

A production-grade FastAPI application featuring dynamic vector database routing, hybrid search (dense + sparse), and automatic cross-encoder re-ranking. This architecture seamlessly integrates retrieval context into a LangChain streaming chat interface, designed to reduce hallucinations and provide observable, context-aware responses.

## Architecture Highlights
* **Dynamic DB Selection:** Route queries on-the-fly to Pinecone, Weaviate, Qdrant, or Milvus using a Factory pattern.
* **Hybrid Search:** Combines semantic search (OpenAI text-embedding-3-small) with keyword search (BM25 or SPLADE) using Reciprocal Rank Fusion (RRF) or alpha-blending.
* **Cross-Encoder Re-ranking:** Utilizes Cohere's re-rank API (rerank-english-v3.0) to refine the candidate pool before LLM injection.
* **Streaming RAG:** Non-blocking context retrieval injected directly into LangChain's asynchronous token stream via Server-Sent Events (SSE).

## Prerequisites
* Python 3.10+
* uv (Fast Python package installer and resolver)
* API keys for OpenAI, Cohere, and your chosen Vector DBs.

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```
   

2. **Set up the environment and install dependencies**
   Using uv for dependency management:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install fastapi uvicorn openai cohere langchain-openai langchain-core pydantic pydantic-settings redis sse-starlette
   
   # Database SDKs & Sparse Encoders
   uv pip install pinecone-client pinecone-text weaviate-client qdrant-client fastembed pymilvus
   ```
   

3. **Configure Environment Variables**
   Create a .env file in the project root:
   ```env
   OPENAI_API_KEY="your-openai-key"
   COHERE_API_KEY="your-cohere-key"

   # Vector DB Credentials (populate those you intend to use)
   PINECONE_API_KEY="your-pinecone-key"
   WEAVIATE_URL="your-weaviate-cluster-url"
   WEAVIATE_API_KEY="your-weaviate-api-key"
   QDRANT_URL="your-qdrant-url"
   QDRANT_API_KEY="your-qdrant-api-key"
   MILVUS_URL="your-milvus-url"
   MILVUS_API_KEY="your-milvus-api-key"
   ```   

## Running the Application

Start the FastAPI server:
```bash
uv run uvicorn app.main:app --reload
```

The API documentation will be available at http://localhost:8000/docs.

## End-to-End Workflow

### Step 1: Provision the Database
Create the necessary schema, dimension config, and indexes for your target provider.
```bash
curl -X POST "http://localhost:8000/api/setup_index" \
-H "Content-Type: application/json" \
-d '{
  "db_provider": "pinecone",
  "index_name": "hybrid-index",
  "dimension": 1536,
  "recreate": true
}'
```


### Step 2: Ingest Documents
Encode (dense + sparse) and upload your corpus.
```bash
curl -X POST "http://localhost:8000/api/ingest" \
-H "Content-Type: application/json" \
-d '{
  "db_provider": "pinecone",
  "index_name": "hybrid-index",
  "dimension" : 1536,
  "documents": [
    "Training a cat requires patience and positive reinforcement.",
    "The feline curled up on the warm carpet beside the window."
  ]
}'
```


### Step 3: Query the RAG Chat Stream
Send a chat request. The backend will perform the hybrid retrieval, re-rank the hits, inject the context into the system prompt, and stream the LLM response back.
```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
-H "Accept: text/event-stream" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": "Based on my documents, what is the best way to train a feline?"
    }
  ],
  "use_rag": true,
  "db_provider": "pinecone",
  "index_name": "hybrid-index",
  "dimension" : 1536,
  "top_k": 10,
  "rerank_top_n": 5
}'
```


## Project Structure
* main.py: FastAPI application routing and ingestion endpoints.
* app/chat.py: Streaming (/chat/stream) and sync (/chat/sync) endpoints with context injection logic.
* app/services/vector_factory.py: Strategy pattern implementation for Pinecone, Weaviate, Qdrant, and Milvus.
* app/services/rag_service.py: Orchestrates the retrieval and re-ranking pipeline.
* app/services/reranker.py: Cohere cross-encoder integration.
* app/services/llm_service.py: LangChain ChatOpenAI generation and streaming wrappers.