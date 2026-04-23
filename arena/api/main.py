import asyncio
import json
import os
import subprocess

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.models import ArenaConfig
from ..core.state_machine import Arena

# ── Config from environment ───────────────────────────────────────────────────

config = ArenaConfig(
    max_agents               = int(os.getenv("MAX_AGENTS",               "8")),
    max_simultaneous_matches = int(os.getenv("MAX_SIMULTANEOUS_MATCHES", "3")),
    map_width                = int(os.getenv("MAP_WIDTH",                "20")),
    map_height               = int(os.getenv("MAP_HEIGHT",               "15")),
    turns_per_match          = int(os.getenv("TURNS_PER_MATCH",          "5")),
    cooldown_seconds         = int(os.getenv("COOLDOWN_SECONDS",         "30")),
    admin_token              = os.getenv("ARENA_ADMIN_TOKEN",    "changeme"),
    azure_openai_endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT",        ""),
    azure_openai_key         = os.getenv("AZURE_OPENAI_KEY",             ""),
    azure_openai_deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT",      "gpt-4o"),
    openai_key               = os.getenv("OPENAI_API_KEY",               ""),
    openai_model             = os.getenv("OPENAI_MODEL",                 "gpt-4o-mini"),
)

arena = Arena(config)

app = FastAPI(title="LatAm Arena", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to omar-florez.github.io after testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_admin(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token != config.admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")

# ── Request models ────────────────────────────────────────────────────────────

class RegisterPayload(BaseModel):
    name:        str
    avatar:      str = "🤖"
    description: str = ""
    endpoint:    str

# ── Public endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "phase": arena.phase}


@app.get("/leaderboard")
def leaderboard():
    return arena._leaderboard_payload()


@app.post("/register")
async def register(payload: RegisterPayload):
    active_count = sum(
        1 for a in arena.agents.values()
        if a.status in ("pending", "active")
    )
    if active_count >= config.max_agents:
        raise HTTPException(status_code=400, detail="Arena is full")

    agent_id = arena.register_agent(
        name=payload.name,
        avatar=payload.avatar,
        description=payload.description,
        endpoint=payload.endpoint,
    )
    # Notify admin subscribers
    await arena.broadcast("agent_pending", {
        "agent_id":    agent_id,
        "name":        payload.name,
        "avatar":      payload.avatar,
        "description": payload.description,
        "endpoint":    payload.endpoint,
    })
    return {
        "agent_id": agent_id,
        "status":   "pending",
        "message":  "Waiting for organizer approval. Stand by.",
    }


@app.get("/stream")
async def stream(request: Request):
    """SSE endpoint consumed by the Phaser visualizer."""
    queue = arena.subscribe()

    async def generate():
        # Send full current state immediately on connect
        snapshot = {"type": "state", "payload": arena.full_state_payload()}
        yield f"data: {json.dumps(snapshot)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        finally:
            arena.unsubscribe(queue)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.get("/admin/agents", dependencies=[Depends(require_admin)])
def admin_list_agents():
    return {
        "agents": [
            {
                "agent_id":    a.agent_id,
                "name":        a.name,
                "avatar":      a.avatar,
                "description": a.description,
                "endpoint":    a.endpoint,
                "status":      a.status,
                "score":       a.score,
                "wins":        a.wins,
                "losses":      a.losses,
            }
            for a in arena.agents.values()
        ],
        "phase": arena.phase,
    }


@app.post("/admin/accept/{agent_id}", dependencies=[Depends(require_admin)])
async def admin_accept(agent_id: str):
    if agent_id not in arena.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    await arena.accept_agent(agent_id)
    return {"status": "accepted", "agent_id": agent_id}


@app.post("/admin/reject/{agent_id}", dependencies=[Depends(require_admin)])
def admin_reject(agent_id: str):
    if agent_id not in arena.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    arena.reject_agent(agent_id)
    return {"status": "rejected", "agent_id": agent_id}


@app.post("/admin/start", dependencies=[Depends(require_admin)])
async def admin_start():
    if arena.phase != "lobby":
        raise HTTPException(status_code=400, detail=f"Tournament is already in phase: {arena.phase}")
    await arena.start_tournament()
    return {"status": "started"}


@app.post("/admin/eliminate/{agent_id}", dependencies=[Depends(require_admin)])
async def admin_eliminate(agent_id: str):
    if agent_id not in arena.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    await arena.eliminate_agent(agent_id)
    return {"status": "eliminated"}


@app.post("/admin/end", dependencies=[Depends(require_admin)])
async def admin_end():
    await arena.end_tournament()
    return {"status": "ended"}


@app.post("/admin/shutdown", dependencies=[Depends(require_admin)])
async def admin_shutdown():
    """Graceful VM shutdown — only call after the event."""
    subprocess.Popen(["sudo", "shutdown", "now"])
    return {"status": "shutting_down"}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arena.api.main:app", host="0.0.0.0", port=8000, reload=False)