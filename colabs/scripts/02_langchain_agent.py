# ============================================================
#  AGENT COLISEUM — BCP Branch — Colab 02: LangChain Agent
# ============================================================
#
#  Before running:
#    1. Click the 🔑 icon in the Colab left sidebar (Secrets)
#    2. Add these secrets:
#         AZURE_API_KEY   → key provided by organizer
#         AZURE_BASE_URL  → https://rsgd15-foundry.openai.azure.com/openai/v1/
#         NGROK_TOKEN     → your ngrok token (ngrok.com)
#    3. Run all cells in order
# ============================================================
#
#  Strategy: LangChain agent with structured memory.
#    - Uses plain list to track match history (no external memory lib)
#    - Uses RunnableSequence (LCEL) for CoT
#    - No external tools during the match — fast responses
#
#  Demonstrates how LangChain abstractions map to the arena API.
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
# !pip install flask flask-cors pyngrok langchain langchain-openai \
#              langchain-community requests -q

# ── CELL 2: Config ───────────────────────────────────────────
import os, json, random
from agent_base import Agent, MatchContext, MatchResult, WorldContext, Position
from agent_server import serve_and_register

from google.colab import userdata
AZURE_API_KEY  = userdata.get('AZURE_API_KEY')
AZURE_BASE_URL = userdata.get('AZURE_BASE_URL')
MODEL          = 'gpt-5'
ARENA_URL      = 'https://agent-coliseum.onrender.com'
NGROK_TOKEN    = userdata.get('NGROK_TOKEN')

# Set env vars so LangChain picks them up automatically
os.environ['OPENAI_API_KEY']  = AZURE_API_KEY
os.environ['OPENAI_BASE_URL'] = AZURE_BASE_URL

# ── CELL 4: LangChain setup ───────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model=MODEL, temperature=0.3)

THINK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a competitive Latin America knowledge agent.
You think carefully before asking or answering.
Always structure your response with these sections:
PLAN: assess match state in 1 sentence.
DRAFT: write your question or answer (1-2 sentences max).
FINAL: the final question or answer only."""),
    ("human", """{input}"""),
])

think_chain = THINK_PROMPT | llm | StrOutputParser()

# ── CELL 5: Agent implementation ─────────────────────────────

class LangChainLatAmAgent(Agent):
    """
    LangChain-powered agent.
    Uses LCEL chain for structured thinking.
    Maintains per-match history as a plain list.
    """

    name        = "LangChain Puma"
    avatar      = "🐆"
    description = "Agente construido con LangChain LCEL y memoria de conversacion"

    def __init__(self):
        self._match_memory   = {}  # match_id → list of {turn, role, summary}
        self._opponent_notes = {}  # opponent_id → {name, result, topic}

    # ── lifecycle ─────────────────────────────────────────────

    def on_arena_start(self, ctx: WorldContext) -> None:
        self._match_memory   = {}
        self._opponent_notes = {}
        print(f"[{self.name}] Arena started with {len(ctx.agents)} agents.")

    def on_match_start(self, ctx: MatchContext) -> None:
        self._match_memory[ctx.match_id] = []
        print(f"[{self.name}] Starting match {ctx.match_id} vs {ctx.opponent_name}")

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        won = result.winner_id == ctx.my_agent_id
        self._opponent_notes[ctx.opponent_agent_id] = {
            "name":   ctx.opponent_name,
            "result": "won" if won else "lost",
            "topic":  ctx.topic,
        }
        print(f"[{self.name}] Match over: {'WON' if won else 'LOST'}")

    # ── world ─────────────────────────────────────────────────

    def move(self, ctx: WorldContext) -> Position:
        dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0),(0,0)])
        return Position(
            x=max(0, min(ctx.map_width  - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    # ── cognition ─────────────────────────────────────────────

    def think(self, ctx: MatchContext) -> str:
        """
        Calls LangChain LCEL chain with structured prompt.
        Uses plain list memory to include conversation history.
        """
        # Build history summary from match history
        history = ""
        for t in ctx.history[-3:]:
            history += (
                f"\nTurn {t['turn_number']}: "
                f"Q={t['question'][:50]} "
                f"A={t['answer'][:50]} "
                f"Score={t['score']}"
            )

        # Local memory summary (what this agent remembers from this match)
        local_mem = self._match_memory.get(ctx.match_id, [])
        mem_summary = ""
        for entry in local_mem[-2:]:
            mem_summary += f"\n  Turn {entry['turn']} ({entry['role']}): {entry['summary']}"

        # Opponent notes from past matches
        opp = self._opponent_notes.get(ctx.opponent_agent_id, {})
        opp_text = (
            f"Previously faced them: {opp.get('result','unknown')} on topic {opp.get('topic','?')}"
            if opp else "First time meeting this opponent."
        )

        input_text = f"""CURRENT MATCH:
Topic: {ctx.topic}
My role: {ctx.role}  |  Turn {ctx.turn}/{ctx.total_turns}
My score: {sum(ctx.my_scores)} pts  |  Opponent: {sum(ctx.opponent_scores)} pts
Opponent: {ctx.opponent_name}
{f'Question to answer: {ctx.current_question}' if ctx.role == 'answerer' else ''}

OPPONENT NOTES: {opp_text}

MATCH HISTORY:{history if history else ' (first turn)'}

MY LOCAL MEMORY:{mem_summary if mem_summary else ' (empty)'}

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

        result = think_chain.invoke({"input": input_text})

        # Store a summary in local memory
        mem = self._match_memory.get(ctx.match_id, [])
        mem.append({
            "turn":    ctx.turn,
            "role":    ctx.role,
            "summary": result[:150],
        })
        self._match_memory[ctx.match_id] = mem

        return result

    def ask(self, ctx: MatchContext) -> str:
        scratchpad = self.think(ctx)
        return self._extract_final(scratchpad)

    def answer(self, ctx: MatchContext) -> str:
        scratchpad = self.think(ctx)
        return self._extract_final(scratchpad)

    def _extract_final(self, text: str) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "FINAL:" in line.upper():
                rest = line.split(":", 1)[1].strip()
                if rest:
                    return rest
                remaining = "\n".join(lines[i+1:]).strip()
                return remaining if remaining else text.split("\n")[-1]
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return text


# ── CELL 5: Run ──────────────────────────────────────────────
from pyngrok import ngrok
ngrok.kill()  # kill any existing tunnels before starting

agent = LangChainLatAmAgent()

serve_and_register(
    agent       = agent,
    arena_url   = ARENA_URL,
    port        = 5001,
    ngrok_token = NGROK_TOKEN,
)
# This cell blocks. The agent is now live and registered.
# Wait for the organizer to accept you in the admin panel.