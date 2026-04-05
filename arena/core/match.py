import asyncio
import uuid
from dataclasses import asdict
import httpx

from .models import (
    MatchContext, MatchResult, Turn,
    AgentInfo, ArenaConfig,
)
from .judge import Judge


class MatchRunner:

    def __init__(self, judge: Judge, config: ArenaConfig):
        self.judge  = judge
        self.config = config

    # ── Public entry point ────────────────────────────────────────────────────

    async def run_match(
        self,
        agent_a:  AgentInfo,
        agent_b:  AgentInfo,
        topic:    str,
        on_event,          # async callable(event_type: str, payload: dict)
    ) -> MatchResult:

        match_id = str(uuid.uuid4())[:8]
        history:    list[Turn] = []
        scores:     dict[str, int] = {
            agent_a.agent_id: 0,
            agent_b.agent_id: 0,
        }
        scratchpads: dict[str, str] = {
            agent_a.agent_id: "",
            agent_b.agent_id: "",
        }

        await on_event("match_start", {
            "match_id":    match_id,
            "topic":       topic,
            "agent_a":     self._agent_summary(agent_a),
            "agent_b":     self._agent_summary(agent_b),
            "total_turns": self.config.turns_per_match,
        })

        for turn_num in range(1, self.config.turns_per_match + 1):

            # Roles alternate each turn
            asker    = agent_a if turn_num % 2 == 1 else agent_b
            answerer = agent_b if turn_num % 2 == 1 else agent_a

            # ── Asker: think → ask ───────────────────────────────────────────
            await on_event("thinking", {
                "turn":       turn_num,
                "agent_id":   asker.agent_id,
                "agent_name": asker.name,
                "role":       "asker",
            })

            asker_ctx = self._build_ctx(
                match_id, topic, turn_num, "asker",
                history, asker, answerer, scores,
                scratchpads[asker.agent_id],
            )
            scratchpad_ask, question = await self._call(asker.endpoint, "ask", asker_ctx)
            scratchpads[asker.agent_id] = scratchpad_ask

            await on_event("question", {
                "turn":        turn_num,
                "asker_id":    asker.agent_id,
                "asker_name":  asker.name,
                "question":    question,
                "scratchpad":  scratchpad_ask,
            })
            await asyncio.sleep(1.5)

            # ── Answerer: think → answer ─────────────────────────────────────
            await on_event("thinking", {
                "turn":       turn_num,
                "agent_id":   answerer.agent_id,
                "agent_name": answerer.name,
                "role":       "answerer",
            })

            answerer_ctx = self._build_ctx(
                match_id, topic, turn_num, "answerer",
                history, answerer, asker, scores,
                scratchpads[answerer.agent_id],
                current_question=question,
            )
            scratchpad_ans, answer = await self._call(answerer.endpoint, "answer", answerer_ctx)
            scratchpads[answerer.agent_id] = scratchpad_ans

            await on_event("answer", {
                "turn":           turn_num,
                "answerer_id":    answerer.agent_id,
                "answerer_name":  answerer.name,
                "answer":         answer,
                "scratchpad":     scratchpad_ans,
            })
            await asyncio.sleep(1.5)

            # ── Judge ────────────────────────────────────────────────────────
            score, reason = self.judge.score(topic, question, answer)
            scores[answerer.agent_id] += score

            turn = Turn(
                turn_number=turn_num,
                asker=asker.agent_id,
                answerer=answerer.agent_id,
                question=question,
                answer=answer,
                score=score,
                score_reason=reason,
                scratchpad_asker=scratchpad_ask,
                scratchpad_answerer=scratchpad_ans,
            )
            history.append(turn)

            await on_event("score", {
                "turn":         turn_num,
                "answerer_id":  answerer.agent_id,
                "score":        score,
                "reason":       reason,
                "running_scores": scores,
            })
            await asyncio.sleep(2.0)

        # ── Determine winner ─────────────────────────────────────────────────
        winner_id = max(scores, key=scores.get)
        loser_id  = min(scores, key=scores.get)

        await on_event("match_end", {
            "match_id":     match_id,
            "winner_id":    winner_id,
            "loser_id":     loser_id,
            "final_scores": scores,
        })

        return MatchResult(
            match_id=match_id,
            winner_id=winner_id,
            loser_id=loser_id,
            turns=history,
            final_scores=scores,
            topic=topic,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_ctx(
        self, match_id, topic, turn, role,
        history, me, opponent, scores, scratchpad,
        current_question="",
    ) -> MatchContext:
        return MatchContext(
            match_id=match_id,
            topic=topic,
            turn=turn,
            total_turns=self.config.turns_per_match,
            role=role,
            history=[self._turn_to_dict(t) for t in history],
            my_agent_id=me.agent_id,
            opponent_agent_id=opponent.agent_id,
            opponent_name=opponent.name,
            my_scores=[t.score for t in history if t.answerer == me.agent_id],
            opponent_scores=[t.score for t in history if t.answerer == opponent.agent_id],
            scratchpad=scratchpad,
            current_question=current_question,
        )

    async def _call(self, endpoint: str, action: str, ctx: MatchContext) -> tuple[str, str]:
        """HTTP call to participant agent. Returns (scratchpad, text)."""
        payload = {
            "match_id":          ctx.match_id,
            "topic":             ctx.topic,
            "turn":              ctx.turn,
            "total_turns":       ctx.total_turns,
            "role":              ctx.role,
            "history":           ctx.history,
            "my_agent_id":       ctx.my_agent_id,
            "opponent_agent_id": ctx.opponent_agent_id,
            "opponent_name":     ctx.opponent_name,
            "my_scores":         ctx.my_scores,
            "opponent_scores":   ctx.opponent_scores,
            "scratchpad":        ctx.scratchpad,
            "current_question":  ctx.current_question,
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(f"{endpoint}/{action}", json=payload)
                data = resp.json()
                return data.get("scratchpad", ""), data.get("text", "[no response]")
        except Exception as e:
            return "", f"[Agent unreachable: {e}]"

    @staticmethod
    def _turn_to_dict(t: Turn) -> dict:
        return {
            "turn_number":  t.turn_number,
            "asker":        t.asker,
            "answerer":     t.answerer,
            "question":     t.question,
            "answer":       t.answer,
            "score":        t.score,
            "score_reason": t.score_reason,
        }

    @staticmethod
    def _agent_summary(a: AgentInfo) -> dict:
        return {"id": a.agent_id, "name": a.name, "avatar": a.avatar}
