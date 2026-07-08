"""
agent_server.py  --  Run this in your Colab or JupyterLab to join the arena.

Usage:
    from agent_server import serve_and_register
    serve_and_register(agent=MyAgent(), arena_url="https://YOUR_ARENA_URL")

Changes from previous version:
    - Uses ngrok CLI directly with --pooling-enabled instead of pyngrok
    - Supports ngrok Pro reserved domains (multiple tunnels on same domain)
    - Falls back to pyngrok if ngrok CLI is not available
"""

import os
import subprocess
import threading
import time

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

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
            text       = agent._extract_final(scratchpad)
        except Exception as e:
            scratchpad, text = f"Error: {e}", f"I'm not sure. (Error: {e})"
        return jsonify({"text": text, "scratchpad": scratchpad})

    @app.route("/answer", methods=["POST"])
    def answer():
        ctx = _parse_match_ctx(request.json)
        try:
            scratchpad = agent.think(ctx)
            text       = agent._extract_final(scratchpad)
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


def _start_ngrok_cli(port: int, ngrok_token: str) -> str:
    """
    Start ngrok tunnel using CLI with --pooling-enabled.
    Required for ngrok Pro accounts with reserved domains.
    """
    subprocess.run(["pkill", "-f", f"ngrok http {port}"], capture_output=True)
    time.sleep(1)

    subprocess.Popen(
        ["ngrok", "http", str(port),
         "--authtoken", ngrok_token,
         "--pooling-enabled",
         "--log", "stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        r       = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = r.json().get("tunnels", [])
        for t in tunnels:
            if str(port) in t.get("config", {}).get("addr", ""):
                return t["public_url"]
        if tunnels:
            return tunnels[0]["public_url"]
    except Exception as e:
        print(f"  Could not get tunnel URL: {e}")
    return None


def _start_pyngrok(port: int, ngrok_token: str) -> str:
    """Fallback: pyngrok. Works on free accounts."""
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = ngrok_token
    tunnel = ngrok.connect(port, "http")
    return tunnel.public_url


def _get_tunnel(port: int, ngrok_token: str) -> str:
    """Try ngrok CLI first (Pro pooling), fall back to pyngrok."""
    try:
        result = subprocess.run(["ngrok", "version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            print("  Using ngrok CLI (--pooling-enabled)")
            url = _start_ngrok_cli(port, ngrok_token)
            if url:
                return url
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("  Using pyngrok (ngrok CLI not found)")
    return _start_pyngrok(port, ngrok_token)


def _reset_arena_if_needed(arena_url: str, admin_token: str) -> None:
    if not admin_token:
        return
    try:
        r     = requests.get(f"{arena_url}/health", timeout=5)
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
    Start the agent server, expose via ngrok, register with the arena.

    Args:
        agent:       Your Agent instance
        arena_url:   The arena backend URL
        port:        Local Flask port (default 5000)
        ngrok_token: Your ngrok auth token (free at ngrok.com)
        admin_token: Arena admin token (optional — resets stale tournaments)
    """
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    time.sleep(1)

    _reset_arena_if_needed(arena_url, admin_token)

    flask_app = _build_flask_app(agent)
    t = threading.Thread(
        target=lambda: flask_app.run(port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    time.sleep(1)

    public_url = _get_tunnel(port, ngrok_token)
    if not public_url:
        print(f"  WARNING: Could not create ngrok tunnel.")
        public_url = f"http://localhost:{port}"

    print(f"\n{'='*55}")
    print(f"  Agent server running at: {public_url}")
    print(f"  Agent name:   {agent.name}")
    print(f"  Agent avatar: {agent.avatar}")
    print(f"{'='*55}\n")

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

    def _keepalive():
        while True:
            time.sleep(30)
            try:
                requests.get(f"{arena_url}/health", timeout=5)
            except Exception:
                pass

    threading.Thread(target=_keepalive, daemon=True).start()

    try:
        t.join()
    except KeyboardInterrupt:
        print(f"\n[{agent.name}] Server stopped.")