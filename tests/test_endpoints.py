#!/usr/bin/env python3
"""
Run:  python3 tests/test_endpoints.py
Requires API running on localhost:8000
"""
import asyncio, json, uuid
import httpx

BASE = "http://localhost:8000"


async def test_health():
    print("\n── Health ────────────────────────────────")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/health/ready")
        print("  ", r.json())
        assert r.json()["checks"]["redis"] == "ok"


async def test_sync():
    print("\n── Sync chat ─────────────────────────────")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "Reply with exactly: HELLO"}],
            "temperature": 0.0,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        print("  Response :", data["message"]["content"])
        print("  Latency  :", data["latency_ms"], "ms")
        print("  Trace-ID :", r.headers.get("X-Trace-ID"))
        return data["session_id"]


async def test_stream():
    print("\n── Stream chat ───────────────────────────")
    tokens = []
    async with httpx.AsyncClient(timeout=60) as c:
        async with c.stream("POST", f"{BASE}/chat/stream", json={
            "messages": [{"role": "user", "content": "Count 1 to 5"}],
        }) as r:
            print("  Session-ID:", r.headers.get("X-Session-ID"))
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    try:
                        ev = json.loads(line[5:])
                        if "token" in ev:
                            tokens.append(ev["token"])
                            print(ev["token"], end="", flush=True)
                    except Exception:
                        pass
    print(f"\n  Tokens received: {len(tokens)}")


async def test_session(session_id: str):
    print("\n── Session continuity ────────────────────")
    async with httpx.AsyncClient(timeout=60) as c:
        await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "My name is Pawan."}],
            "session_id": session_id, "temperature": 0.0,
        })
        r2 = await c.post(f"{BASE}/chat/sync", json={
            "messages": [{"role": "user", "content": "What is my name?"}],
            "session_id": session_id, "temperature": 0.0,
        })
        print("  ", r2.json()["message"]["content"])


async def main():
    print("=" * 50)
    print("  FastAPI Chat — Integration Tests")
    print("=" * 50)
    await test_health()
    sid = await test_sync()
    await test_stream()
    await test_session(sid)
    print("\n\n✅  All tests passed")

if __name__ == "__main__":
    asyncio.run(main())
