import asyncio
import random
import time
import uuid
from typing import Callable

from .models import (
    ArenaPhase, AgentStatus,
    AgentInfo, Position, ArenaConfig, MatchResult,
)
from .judge import Judge
from .match import MatchRunner

TOPICS = [
    "Latin American geography and natural wonders",
    "Pre-Columbian civilizations and indigenous cultures",
    "Latin American independence and revolutionary history",
    "Latin American literature and Nobel Prize winners",
    "Latin American music, art, and cultural traditions",
    "Latin American economics and natural resources",
    "Modern Latin American politics and social movements",
    "Latin American science, technology, and sports",
]


class Arena:

    def __init__(self, config: ArenaConfig):
        self.config = config
        self.phase: ArenaPhase = ArenaPhase.LOBBY

        self.agents:    dict[str, AgentInfo] = {}
        self._active_matches: set[str]       = set()   # agent_ids in a match
        self._cooldowns: dict[str, float]    = {}       # agent_id → timestamp
        self._subscribers: list              = []       # SSE queues
        self._roaming_task                   = None     # cancellable task ref

        judge        = Judge(
                             azure_endpoint   = config.azure_openai_endpoint,
                             azure_key        = config.azure_openai_key,
                             azure_deployment = config.azure_openai_deployment,
                             openai_key       = config.openai_key,
                             openai_model     = config.openai_model,
                         )
        self.runner  = MatchRunner(judge, config)

    # ── SSE pub/sub ───────────────────────────────────────────────────────────

    async def broadcast(self, event_type: str, payload: dict):
        for q in list(self._subscribers):
            await q.put({"type": event_type, "payload": payload})

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    # ── Agent registry ────────────────────────────────────────────────────────

    def register_agent(self, name: str, avatar: str,
                       description: str, endpoint: str) -> str:
        agent_id = str(uuid.uuid4())[:8]
        self.agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            name=name,
            avatar=avatar,
            description=description,
            endpoint=endpoint,
            status=AgentStatus.PENDING,
            position=Position(x=0, y=0),
        )
        return agent_id

    async def accept_agent(self, agent_id: str):
        a = self.agents[agent_id]
        a.status   = AgentStatus.ACTIVE
        a.position = self._spawn_position()
        await self.broadcast("agent_accepted", {
            "agent_id":    a.agent_id,
            "name":        a.name,
            "avatar":      a.avatar,
            "description": a.description,
            "position":    {"x": a.position.x, "y": a.position.y},
        })

    def reject_agent(self, agent_id: str):
        self.agents.pop(agent_id, None)

    async def eliminate_agent(self, agent_id: str):
        a = self.agents.get(agent_id)
        if not a:
            return
        a.status = AgentStatus.ELIMINATED
        await self.broadcast("agent_eliminated", {
            "agent_id": agent_id,
            "name":     a.name,
            "avatar":   a.avatar,
        })
        # Notify participant's Colab
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{a.endpoint}/eliminated")
        except Exception:
            pass

    # ── Tournament lifecycle ──────────────────────────────────────────────────

    async def start_tournament(self):
        self.phase = ArenaPhase.ROAMING
        await self.broadcast("phase_change", {"phase": "roaming"})
        self._roaming_task = asyncio.create_task(self._roaming_loop())

    async def end_tournament(self):
        self.phase = ArenaPhase.ENDED
        # Cancel the roaming loop immediately
        if self._roaming_task and not self._roaming_task.done():
            self._roaming_task.cancel()
        active = self._active_agents()
        winner = max(active, key=lambda a: a.score) if active else None
        await self.broadcast("tournament_ended", {
            "winner": {
                "agent_id": winner.agent_id,
                "name":     winner.name,
                "avatar":   winner.avatar,
                "score":    winner.score,
            } if winner else None,
            "leaderboard": self._leaderboard_payload(),
        })

    # ── Roaming loop ──────────────────────────────────────────────────────────

    async def _roaming_loop(self):
        while self.phase in (ArenaPhase.ROAMING, ArenaPhase.FINALS):
            await asyncio.sleep(2)
            active = self._active_agents()

            if not active:
                await self.end_tournament()
                return

            if len(active) == 1:
                await self.end_tournament()
                return

            # Transition to FINALS when two agents remain
            if len(active) == 2 and self.phase == ArenaPhase.ROAMING:
                self.phase = ArenaPhase.FINALS
                await self.broadcast("phase_change", {"phase": "finals"})

            # Stop immediately if phase changed
            if self.phase not in (ArenaPhase.ROAMING, ArenaPhase.FINALS):
                return

            # Move every active agent toward closest opponent
            for agent in active:
                if agent.agent_id in self._active_matches:
                    continue   # skip agents already in a match
                new_pos = self._step_toward(agent, active)
                agent.position = new_pos
                await self.broadcast("agent_moved", {
                    "agent_id": agent.agent_id,
                    "position": {"x": new_pos.x, "y": new_pos.y},
                })

            # Check for proximity matches
            if len(self._active_matches) < self.config.max_simultaneous_matches:
                await self._check_proximity(active)

    async def _check_proximity(self, active: list[AgentInfo]):
        now = time.time()
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                # Skip if either is already in a match or on cooldown
                if a.agent_id in self._active_matches: continue
                if b.agent_id in self._active_matches: continue
                if now - self._cooldowns.get(a.agent_id, 0) < self.config.cooldown_seconds: continue
                if now - self._cooldowns.get(b.agent_id, 0) < self.config.cooldown_seconds: continue
                if self._adjacent(a.position, b.position):
                    asyncio.create_task(self._run_match(a, b))
                    break  # one match trigger per tick

    async def _run_match(self, agent_a: AgentInfo, agent_b: AgentInfo):
        self._active_matches.add(agent_a.agent_id)
        self._active_matches.add(agent_b.agent_id)
        topic = random.choice(TOPICS)

        try:
            result: MatchResult = await self.runner.run_match(
                agent_a, agent_b, topic, on_event=self.broadcast
            )
            # Update tournament scores
            self.agents[result.winner_id].score += result.final_scores[result.winner_id]
            self.agents[result.winner_id].wins  += 1
            self.agents[result.loser_id].losses += 1

            # Eliminate the loser
            await self.eliminate_agent(result.loser_id)

            # Broadcast updated leaderboard
            await self.broadcast("leaderboard_update", self._leaderboard_payload())

        finally:
            self._active_matches.discard(agent_a.agent_id)
            self._active_matches.discard(agent_b.agent_id)
            now = time.time()
            self._cooldowns[agent_a.agent_id] = now
            self._cooldowns[agent_b.agent_id] = now

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _active_agents(self) -> list[AgentInfo]:
        return [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]

    def _spawn_position(self) -> Position:
        return Position(
            x=random.randint(1, self.config.map_width  - 2),
            y=random.randint(1, self.config.map_height - 2),
        )

    def _step_toward(self, agent: AgentInfo, others: list) -> Position:
        """Move one tile toward the closest other active agent."""
        others = [a for a in others if a.agent_id != agent.agent_id
                  and a.agent_id not in self._active_matches]
        if not others:
            return agent.position

        # Find closest
        target = min(others, key=lambda a: abs(a.position.x - agent.position.x)
                                         + abs(a.position.y - agent.position.y))

        dx = 0
        dy = 0
        if target.position.x > agent.position.x:   dx =  1
        elif target.position.x < agent.position.x: dx = -1
        if target.position.y > agent.position.y:   dy =  1
        elif target.position.y < agent.position.y: dy = -1

        # Move along the longer axis first for direct pathing
        ax = abs(target.position.x - agent.position.x)
        ay = abs(target.position.y - agent.position.y)
        if ax >= ay:
            dy = 0
        else:
            dx = 0

        return Position(
            x=max(0, min(self.config.map_width  - 1, agent.position.x + dx)),
            y=max(0, min(self.config.map_height - 1, agent.position.y + dy)),
        )

    def _adjacent(self, a: Position, b: Position) -> bool:
        return abs(a.x - b.x) <= 2 and abs(a.y - b.y) <= 2

    def _leaderboard_payload(self) -> dict:
        sorted_agents = sorted(
            self.agents.values(), key=lambda a: a.score, reverse=True
        )
        return {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "name":     a.name,
                    "avatar":   a.avatar,
                    "score":    a.score,
                    "wins":     a.wins,
                    "losses":   a.losses,
                    "status":   a.status,
                }
                for a in sorted_agents
            ]
        }

    def full_state_payload(self) -> dict:
        """Sent to new SSE subscribers immediately on connect."""
        return {
            "phase":  self.phase,
            "agents": [
                {
                    "agent_id":    a.agent_id,
                    "name":        a.name,
                    "avatar":      a.avatar,
                    "description": a.description,
                    "position":    {"x": a.position.x, "y": a.position.y},
                    "score":       a.score,
                    "wins":        a.wins,
                    "losses":      a.losses,
                    "status":      a.status,
                }
                for a in self.agents.values()
            ],
        }