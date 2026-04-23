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

from pyngrok import ngrok
ngrok.kill()  # kills all existing tunnels on this account
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
            prompt = f"""You are competing in a Latin America knowledge contest.
Topic: {ctx.topic}
Role: asker

Think step by step using this reasoning structure:

# SITUATION  [Chain-of-Thought — Wei et al. 2022, NeurIPS]
# Decompose the current state before acting.
# "chain of thought prompting significantly improves the ability
#  of large language models to perform complex reasoning."
# → State the topic, your role, turn number, and current scores.

SITUATION: <assess topic, role, turn, score gap>

# OPPONENT  [Theory of Mind / Opponent Modeling — Langner et al. 2023]
# Model what your opponent knows and how they reason.
# ToM-enabled agents outperform reactive agents in competitive settings
# by anticipating the opponent's next move before it happens.
# → What are their known weaknesses? What topics did they struggle with?

OPPONENT: <model their knowledge gaps and tendencies>

# GOAL  [ReAct — Yao et al. 2022, ICLR 2023]
# Interleave reasoning with goal-directed action planning.
# "ReAct generates verbal reasoning traces and task-specific actions
#  in an interleaved manner."
# → State your concrete objective for this specific turn.

GOAL: <one concrete objective for this turn>

# DRAFT  [Scratchpad — Nye et al. 2021]
# Use an intermediate scratchpad to produce a first attempt
# before committing to a final answer.
# "Scratchpads allow models to show their work, dramatically
#  improving accuracy on multi-step reasoning tasks."
# → Write your first attempt at the question or answer.

DRAFT: <first attempt — question if asker, answer if answerer>

# CRITIQUE  [Self-Refine — Madaan et al. 2023, NeurIPS]
# Iteratively improve output using self-generated feedback.
# "Self-Refine produces significantly better outputs than
#  one-step generation across diverse tasks."
# → Is the draft good enough? Too vague? Missing a key fact?

CRITIQUE: <evaluate draft quality, identify gaps>

# FINAL  [Reflexion — Shinn et al. 2023]
# Commit to a revised response informed by self-reflection.
# "Reflexion agents verbally reflect on task feedback signals
#  to maintain an episodic memory buffer."
# → Write the final polished question (1 sentence) or answer (1-2 sentences).

FINAL: <final question (1 sentence) or answer (1-2 sentences max, be concise)>
"""
        else:
            prompt = f"""You are competing in a Latin America knowledge contest.
Topic: {ctx.topic}
Role: answerer
Question to answer: {ctx.current_question}

Think step by step using this reasoning structure:

# SITUATION  [Chain-of-Thought — Wei et al. 2022, NeurIPS]
# Decompose the current state before acting.
# "chain of thought prompting significantly improves the ability
#  of large language models to perform complex reasoning."
# → State the topic, your role, turn number, and current scores.

SITUATION: <assess topic, role, turn, score gap>

# OPPONENT  [Theory of Mind / Opponent Modeling — Langner et al. 2023]
# Model what your opponent knows and how they reason.
# ToM-enabled agents outperform reactive agents in competitive settings
# by anticipating the opponent's next move before it happens.
# → What are their known weaknesses? What topics did they struggle with?

OPPONENT: <model their knowledge gaps and tendencies>

# GOAL  [ReAct — Yao et al. 2022, ICLR 2023]
# Interleave reasoning with goal-directed action planning.
# "ReAct generates verbal reasoning traces and task-specific actions
#  in an interleaved manner."
# → State your concrete objective for this specific turn.

GOAL: <one concrete objective for this turn>

# DRAFT  [Scratchpad — Nye et al. 2021]
# Use an intermediate scratchpad to produce a first attempt
# before committing to a final answer.
# "Scratchpads allow models to show their work, dramatically
#  improving accuracy on multi-step reasoning tasks."
# → Write your first attempt at the question or answer.

DRAFT: <first attempt — question if asker, answer if answerer>

# CRITIQUE  [Self-Refine — Madaan et al. 2023, NeurIPS]
# Iteratively improve output using self-generated feedback.
# "Self-Refine produces significantly better outputs than
#  one-step generation across diverse tasks."
# → Is the draft good enough? Too vague? Missing a key fact?

CRITIQUE: <evaluate draft quality, identify gaps>

# FINAL  [Reflexion — Shinn et al. 2023]
# Commit to a revised response informed by self-reflection.
# "Reflexion agents verbally reflect on task feedback signals
#  to maintain an episodic memory buffer."
# → Write the final polished question (1 sentence) or answer (1-2 sentences).

FINAL: <final question (1 sentence) or answer (1-2 sentences max, be concise)>
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
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