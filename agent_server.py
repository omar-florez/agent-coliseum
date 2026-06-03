"""
agent_server.py  --  Run this in your Colab to join the arena.

Usage:
    from agent_server import serve_and_register
    serve_and_register(agent=MyAgent(), arena_url="https://YOUR_ARENA_URL")
"""
import json
import threading
from dataclasses import dataclass

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

from agent_base import (
    Agent, MatchContext, MatchResult, WorldContext,
    AgentInfo, Position
)


def _parse_match_ctx(data: dict) -> MatchContext:
    return MatchContext(
        match_id          = data["match_id"],
        topic             = data["topic"],
        turn              = data["turn"],
        total_turns       = data["total_turns"],
        role              = data["role"],
        history           = data["history"],
        my_agent_id       = data["my_agent_id"],
        opponent_agent_id = data["opponent_agent_id"],
        opponent_name     = data["opponent_name"],
        my_scores         = data["my_scores"],
        opponent_scores   = data["opponent_scores"],
        scratchpad        = data.get("scratchpad", ""),
        current_question  = data.get("current_question", ""),
    )


def _parse_world_ctx(data: dict) -> WorldContext:
    agents = [
        AgentInfo(
            agent_id = a["agent_id"],
            name     = a["name"],
            avatar   = a["avatar"],
            score    = a["score"],
            position = Position(x=a["position"]["x"], y=a["position"]["y"]),
            status   = a["status"],
        )
        for a in data.get("agents", [])
    ]
    return WorldContext(
        phase       = data["phase"],
        my_agent_id = data["my_agent_id"],
        my_position = Position(x=data["my_position"]["x"], y=data["my_position"]["y"]),
        my_score    = data["my_score"],
        agents      = agents,
        map_width   = data["map_width"],
        map_height  = data["map_height"],
    )


def _build_flask_app(agent: Agent) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route("/ask", methods=["POST"])
    def ask():
        ctx = _parse_match_ctx(request.json)
        try:
            scratchpad = agent.think(ctx)
            text       = agent.ask(ctx)
        except Exception as e:
            scratchpad, text = f"Error: {e}", f"I'm not sure. (Error: {e})"
        return jsonify({"text": text, "scratchpad": scratchpad})

    @app.route("/answer", methods=["POST"])
    def answer():
        ctx = _parse_match_ctx(request.json)
        try:
            scratchpad = agent.think(ctx)
            text       = agent.answer(ctx)
        except Exception as e:
            scratchpad, text = f"Error: {e}", f"I'm not sure. (Error: {e})"
        return jsonify({"text": text, "scratchpad": scratchpad})

    @app.route("/move", methods=["POST"])
    def move():
        ctx = _parse_world_ctx(request.json)
        try:
            pos = agent.move(ctx)
        except Exception:
            import random
            pos = Position(x=random.randint(0, 19), y=random.randint(0, 14))
        return jsonify({"x": pos.x, "y": pos.y})

    @app.route("/should_challenge", methods=["POST"])
    def should_challenge():
        data   = request.json
        ctx    = _parse_world_ctx(data["ctx"])
        target = AgentInfo(
            agent_id = data["target"]["agent_id"],
            name     = data["target"]["name"],
            avatar   = data["target"]["avatar"],
            score    = data["target"]["score"],
            position = Position(x=data["target"]["position"]["x"],
                                y=data["target"]["position"]["y"]),
            status   = data["target"]["status"],
        )
        try:
            result = agent.should_challenge(ctx, target)
        except Exception:
            result = True
        return jsonify({"challenge": result})

    @app.route("/on_match_start", methods=["POST"])
    def on_match_start():
        ctx = _parse_match_ctx(request.json)
        try:
            agent.on_match_start(ctx)
        except Exception:
            pass
        return jsonify({"ok": True})

    @app.route("/on_match_end", methods=["POST"])
    def on_match_end():
        data   = request.json
        ctx    = _parse_match_ctx(data["ctx"])
        result = MatchResult(
            match_id     = data["result"]["match_id"],
            winner_id    = data["result"]["winner_id"],
            loser_id     = data["result"]["loser_id"],
            turns        = data["result"]["turns"],
            final_scores = data["result"]["final_scores"],
            topic        = data["result"]["topic"],
        )
        try:
            agent.on_match_end(ctx, result)
        except Exception:
            pass
        return jsonify({"ok": True})

    @app.route("/eliminated", methods=["POST"])
    def eliminated():
        try:
            agent.on_eliminated()
        except Exception:
            pass
        return jsonify({"ok": True})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "name": agent.name, "avatar": agent.avatar})

    return app


def _reset_arena_if_needed(arena_url: str, admin_token: str) -> None:
    """
    Reset the arena to lobby if it is not already there.
    This clears stale agents from previous runs.
    Safe to call from any agent — only resets if tournament
    is already running or ended (not if in lobby waiting for players).
    """
    import requests

    if not admin_token:
        return

    try:
        r = requests.get(f"{arena_url}/health", timeout=5)
        phase = r.json().get("phase", "lobby")
        if phase != "lobby":
            reset = requests.post(
                f"{arena_url}/admin/reset",
                headers={"Authorization": f"Bearer {admin_token}"},
                timeout=5,
            )
            print(f"[Arena] Phase was '{phase}' — reset to lobby: {reset.status_code}")
        else:
            print(f"[Arena] Already in lobby — no reset needed")
    except Exception as e:
        print(f"[Arena] Could not check/reset arena: {e}")


def serve_and_register(
    agent:       Agent,
    arena_url:   str,
    port:        int  = 5000,
    ngrok_token: str  = None,
    admin_token: str  = None,
):
    """
    Start the agent server, expose it via ngrok, register with the arena.
    Automatically resets the arena if a previous tournament is still running.

    Args:
        agent:       Your Agent instance
        arena_url:   The arena backend URL
        port:        Local Flask port (default 5000)
        ngrok_token: Your ngrok auth token (free at ngrok.com)
        admin_token: Arena admin token — if provided, resets stale tournaments
    """
    import requests
    import subprocess

    # Kill any existing process on this port
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)

    # Reset arena if needed
    _reset_arena_if_needed(arena_url, admin_token)

    # Configure ngrok
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)

    flask_app = _build_flask_app(agent)

    # Start Flask in background thread
    t = threading.Thread(
        target=lambda: flask_app.run(port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()

    # Open ngrok tunnel
    tunnel     = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    print(f"\n{'='*55}")
    print(f"  Agent server running at: {public_url}")
    print(f"  Agent name:   {agent.name}")
    print(f"  Agent avatar: {agent.avatar}")
    print(f"{'='*55}\n")

    # Register with the arena
    try:
        resp = requests.post(f"{arena_url}/register", json={
            "name":        agent.name,
            "avatar":      agent.avatar,
            "description": agent.description,
            "endpoint":    public_url,
        }, timeout=10)
        data = resp.json()
        print(f"  Status:   {data['status']}")
        print(f"  Agent ID: {data['agent_id']}")
        print(f"  {data['message']}")
        print(f"\n  Waiting for organizer to accept you into the arena...\n")
    except Exception as e:
        print(f"  Could not reach arena at {arena_url}: {e}")
        print(f"  Your server is running -- try registering manually.")

    # Keepalive thread — pings arena every 30s to prevent ngrok tunnel drops
    def _keepalive():
        import time
        while True:
            time.sleep(30)
            try:
                requests.get(f"{arena_url}/health", timeout=5)
            except Exception:
                pass

    threading.Thread(target=_keepalive, daemon=True).start()

    # Keep main thread alive
    try:
        t.join()
    except KeyboardInterrupt:
        print(f"\n[{agent.name}] Server stopped.")
        ngrok.disconnect(public_url)
