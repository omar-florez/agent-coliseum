# ============================================================
#  LATAM ARENA — Colab 03: Naive Baseline Agent
# ============================================================
#
#  Strategy: The simplest possible agent.
#    - think()   → one-shot prompt, no context, no memory
#    - ask()     → asks a random question, no strategy
#    - answer()  → answers without any retrieved knowledge
#    - move()    → pure random walk
#    - memory    → none (every turn is independent)
#
#  PURPOSE: Demonstrate during the talk WHY memory + RAG matters.
#  This agent will visibly perform worse than Colabs 01 and 02.
#
#  Compatible with: OpenAI, Anthropic, or any LLM provider.
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
# !pip install flask flask-cors pyngrok openai requests -q

# ── CELL 2: Config ───────────────────────────────────────────
import random
from openai import OpenAI
from agent_base import Agent, MatchContext, MatchResult, WorldContext, Position
from agent_server import serve_and_register

OPENAI_API_KEY = "sk-..."
ARENA_URL      = "https://agent-coliseum.onrender.com"
NGROK_TOKEN    = "your_ngrok_token"

client = OpenAI(api_key=OPENAI_API_KEY)

# ── CELL 3: Agent implementation ─────────────────────────────

class NaiveAgent(Agent):
    """
    The baseline naive agent.

    No memory, no RAG, no strategy.
    Just raw GPT with minimal context.

    Used during the talk to show the gap between
    naive and agentic approaches.
    """

    name        = "Bot Básico"
    avatar      = "🤖"
    description = "Un agente simple sin memoria ni recuperación de conocimiento"

    # ── lifecycle (all default no-ops) ────────────────────────
    # on_arena_start, on_match_start, on_match_end: default empty

    # ── world (all random) ────────────────────────────────────
    # move(): default random walk (from Agent base class)
    # should_challenge(): default always True

    # ── cognition ─────────────────────────────────────────────

    def think(self, ctx: MatchContext) -> str:
        """
        Minimal one-shot prompt.
        No history, no memory, no RAG.
        """
        if ctx.role == "asker":
            prompt = f"You are competing in a Latin America knowledge contest. Topic: {ctx.topic}. Ask one challenging question in Spanish or English. Just the question, nothing else."
        else:
            prompt = f"You are competing in a Latin America knowledge contest. Topic: {ctx.topic}. Question: {ctx.current_question}. Answer as accurately as you can in 1-3 sentences."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    def ask(self, ctx: MatchContext) -> str:
        return self.think(ctx)

    def answer(self, ctx: MatchContext) -> str:
        return self.think(ctx)


# ── CELL 4: Run ──────────────────────────────────────────────
agent = NaiveAgent()

serve_and_register(
    agent       = agent,
    arena_url   = ARENA_URL,
    port        = 5002,       # different port from other Colabs
    ngrok_token = NGROK_TOKEN,
)

# ──────────────────────────────────────────────────────────────
#
#  TALK NARRATIVE — why this agent loses:
#
#  Turn 1:  Doesn't know what the opponent asked before.
#           Might ask the same topic twice (no history).
#
#  Turn 3:  Doesn't remember its own scratchpad.
#           Might contradict itself across turns.
#
#  Turn 5:  No RAG → answers from training data only.
#           Misses obscure LatAm facts that Cóndor RAG retrieves instantly.
#
#  Bottom line: same LLM, same API, radically different performance.
#  That's what agency buys you.
#
# ──────────────────────────────────────────────────────────────
