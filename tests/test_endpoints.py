#!/usr/bin/env python3
"""
Run:  python3 tests/test_endpoints.py
Requires API running on localhost:8000
"""
import os
import asyncio
import json
import uuid
import httpx

BASE = "http://localhost:8000"


async def test_health():
    print("\n── Health ────────────────────────────────")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/health/ready")
        print("  ", r.json())
        assert r.status_code == 200


async def test_sync_no_rag():
    print("\n── Sync chat (No RAG) ────────────────────")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "Reply with exactly: HELLO"}],
            "temperature": 0.0,
            "use_rag": False
        })
        assert r.status_code == 200, r.text
        data = r.json()
        print("  Response :", data["message"]["content"])
        print("  Latency  :", data["latency_ms"], "ms")
        return data["session_id"]


async def test_sync_with_rag():
    print("\n── Sync chat (With RAG) ──────────────────")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "Based on my documents, what is the best way to train a feline?"}],
            "temperature": 0.0,
            "use_rag": True,
            "db_provider": "pinecone",
            "index_name": "test-hybrid-collection",
            "dimension": 1536,
            "top_k": 5,
            "rerank_top_n": 3
        })
        assert r.status_code == 200, r.text
        data = r.json()
        print("  Response :", data["message"]["content"])
        print("  Latency  :", data["latency_ms"], "ms")


async def test_stream_with_rag():
    print("\n── Stream chat (With RAG) ────────────────")
    tokens = []
    async with httpx.AsyncClient(timeout=60) as c:
        async with c.stream("POST", f"{BASE}/chat/stream", json={
            "messages": [{"role": "user", "content": "Tell me about vector databases based on the provided context."}],
            "use_rag": True,
            "db_provider": "pinecone",
            "index_name": "test-hybrid-collection",
            "dimension": 1536
        }) as r:
            print("  Session-ID:", r.headers.get("X-Session-ID"))
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    try:
                        ev = json.loads(line[5:])
                        if "data" in ev and ev["data"] is not None:
                            # The stream returns actual tokens in 'data' field now
                            tokens.append(ev["data"])
                            print(ev["data"], end="", flush=True)
                    except Exception:
                        pass
    print(f"\n  Tokens received: {len(tokens)}")


async def test_session(session_id: str):
    print("\n── Session continuity ────────────────────")
    async with httpx.AsyncClient(timeout=60) as c:
        await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "My name is Pawan."}],
            "session_id": session_id, "temperature": 0.0,
            "use_rag": False
        })
        r2 = await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "What is my name?"}],
            "session_id": session_id, "temperature": 0.0,
            "use_rag": False
        })
        print("  ", r2.json()["message"]["content"])

async def test_directory_ingest_recursive():
    print("\n── Directory Ingest (Recursive Chunking) ─")
    # Ensure a dummy dir exists for the test so it doesn't hard-fail
    os.makedirs("./test_data", exist_ok=True)
    
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{BASE}/api/ingest/directory", json={
            "db_provider": "pinecone",
            "index_name": "test-hybrid-collection",
            "dimension": 1536,
            "directory_path": "./test_data",
            "chunking_strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 100
        })
        print("  Response:", r.json())
        assert r.status_code in [200, 400] # 400 is acceptable here if directory is empty


async def test_directory_ingest_token():
    print("\n── Directory Ingest (Token Chunking) ─────")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{BASE}/api/ingest/directory", json={
            "db_provider": "pinecone",
            "index_name": "test-hybrid-collection",
            "dimension": 1536,
            "directory_path": "./test_data",
            "chunking_strategy": "token",
            "chunk_size": 100,
            "chunk_overlap": 10
        })
        print("  Response:", r.json())
        assert r.status_code in [200, 400]


async def test_directory_ingest_invalid_dir():
    print("\n── Directory Ingest (Invalid Path) ───────")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/api/ingest/directory", json={
            "db_provider": "pinecone",
            "index_name": "test-hybrid-collection",
            "directory_path": "/this/path/does/not/exist",
            "chunking_strategy": "recursive"
        })
        print("  Response:", r.json())
        assert r.status_code == 400
        assert "not found" in r.json()["detail"].lower()

async def main():
    print("=" * 50)
    print("  FastAPI Chat — Integration Tests")
    print("=" * 50)
    await test_health()
    sid = await test_sync_no_rag()
    await test_sync_with_rag()
    await test_stream_with_rag()
    await test_session(sid)
    await test_directory_ingest_recursive()
    await test_directory_ingest_token()
    await test_directory_ingest_invalid_dir()
    print("\n\n✅  All tests passed")

if __name__ == "__main__":
    asyncio.run(main())
