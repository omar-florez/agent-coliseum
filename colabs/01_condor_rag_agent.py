# ============================================================
#  LATAM ARENA — Colab 01: Custom OpenAI Agent + RAG
# ============================================================
#
#  Strategy: Full agentic agent.
#    - think()   → structured 6-step CoT prompt via OpenAI
#    - ask()     → strategic question targeting opponent gaps
#    - answer()  → RAG-augmented answer (FAISS + sentence-transformers)
#    - move()    → aggressive: seek weakest opponent
#    - memory    → tracks opponent topics and scores across matches
#
#  Recommended model: gpt-4o-mini (cheap) or gpt-4o (best)
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
# !pip install flask flask-cors pyngrok openai \
#              sentence-transformers faiss-cpu requests -q

# ── CELL 2: Imports ──────────────────────────────────────────
import os, json, random
from openai import OpenAI

# Copy agent_base.py and agent_server.py here or upload them
# For the talk: they are pre-installed in the Colab environment
from agent_base import Agent, MatchContext, MatchResult, WorldContext, Position
from agent_server import serve_and_register

# ── CELL 3: Config ───────────────────────────────────────────
OPENAI_API_KEY = "sk-..."        # YOUR OpenAI key
ARENA_URL      = "https://agent-coliseum.onrender.com"
NGROK_TOKEN    = "your_ngrok_token"    # free at ngrok.com

client = OpenAI(api_key=OPENAI_API_KEY)

# ── CELL 4: RAG setup ────────────────────────────────────────
# Build a FAISS index from latam_facts.jsonl
# Upload latam_facts.jsonl to this Colab or fetch from HuggingFace Hub

def build_rag_index(facts_path="latam_facts.jsonl"):
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    facts = []
    with open(facts_path) as f:
        for line in f:
            if line.strip():
                facts.append(json.loads(line))

    texts = [f["text"] for f in facts]
    print(f"Encoding {len(texts)} facts…")
    embeddings = model.encode(texts, show_progress_bar=True,
                              batch_size=64, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    print(f"RAG index ready: {index.ntotal} facts")
    return model, index, facts


def search_rag(query: str, top_k: int = 3) -> list[str]:
    """Return top-k fact strings relevant to query."""
    vec = rag_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = rag_index.search(vec, top_k)
    return [rag_facts[i]["text"] for i in ids[0] if i >= 0]


# Build index (takes ~30s first time, free on Colab CPU)
rag_model, rag_index, rag_facts = build_rag_index()

# ── CELL 5: Agent implementation ─────────────────────────────

class CondorAgent(Agent):
    """
    Full agentic agent:
    - RAG-augmented answers (free local embeddings + FAISS)
    - 6-step chain-of-thought via OpenAI
    - Opponent memory: tracks topics and scoring patterns
    - Strategic movement: seeks weakest opponent
    """

    name        = "Cóndor RAG"
    avatar      = "🦅"
    description = "Agente con memoria de oponentes y búsqueda semántica de hechos latinoamericanos"

    def __init__(self):
        self._memory = {}        # opponent_id → {topics_asked, avg_score, wins, losses}
        self._my_topics = []     # topics I've seen so far

    # ── lifecycle ────────────────────────────────────────────

    def on_arena_start(self, ctx: WorldContext) -> None:
        self._memory = {}
        print(f"[{self.name}] Tournament started. {len(ctx.agents)} agents on map.")

    def on_match_start(self, ctx: MatchContext) -> None:
        opp = ctx.opponent_agent_id
        if opp not in self._memory:
            self._memory[opp] = {
                "name": ctx.opponent_name,
                "topics_failed": [],
                "avg_score": 5.0,
                "wins": 0,
                "losses": 0,
            }
        print(f"[{self.name}] Match vs {ctx.opponent_name} on topic: {ctx.topic}")

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        opp = ctx.opponent_agent_id
        won = result.winner_id == ctx.my_agent_id
        if opp in self._memory:
            if won: self._memory[opp]["wins"] += 1
            else:   self._memory[opp]["losses"] += 1
        print(f"[{self.name}] Match ended. {'WON' if won else 'LOST'}.")

    def on_eliminated(self) -> None:
        print(f"[{self.name}] Eliminated. Final memory: {self._memory}")

    # ── world strategy ───────────────────────────────────────

    def move(self, ctx: WorldContext) -> Position:
        """Move toward the opponent with the lowest score (easiest target)."""
        active = [a for a in ctx.agents
                  if a.status == "active" and a.agent_id != ctx.my_agent_id]
        if not active:
            return self._random_move(ctx)

        # Find weakest
        target = min(active, key=lambda a: a.score)
        dx = 1 if target.position.x > ctx.my_position.x else -1 if target.position.x < ctx.my_position.x else 0
        dy = 1 if target.position.y > ctx.my_position.y else -1 if target.position.y < ctx.my_position.y else 0
        return Position(
            x=max(0, min(ctx.map_width  - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    def should_challenge(self, ctx: WorldContext, target) -> bool:
        """Avoid challenging the strongest agent unless it's the finals."""
        active = [a for a in ctx.agents if a.status == "active"]
        if len(active) <= 2:  # finals: always fight
            return True
        strongest = max(active, key=lambda a: a.score) if active else None
        return target.agent_id != (strongest.agent_id if strongest else None)

    # ── cognition ────────────────────────────────────────────

    def think(self, ctx: MatchContext) -> str:
        """6-step structured CoT. RAG-augmented context injected."""
        # Retrieve relevant facts for this topic
        rag_hits = search_rag(ctx.topic + " " + ctx.current_question, top_k=3)
        rag_context = "\n".join(f"- {f}" for f in rag_hits)

        # Opponent memory summary
        opp_mem = self._memory.get(ctx.opponent_agent_id, {})
        opp_summary = (
            f"Known info: {opp_mem.get('wins', 0)} wins, {opp_mem.get('losses', 0)} losses. "
            f"Topics they struggled with: {opp_mem.get('topics_failed', [])}"
            if opp_mem else "No prior history with this opponent."
        )

        # Turn history summary
        history_text = ""
        for t in ctx.history[-3:]:
            history_text += (
                f"\n  Turn {t['turn_number']}: "
                f"Q={t['question'][:60]} A={t['answer'][:60]} Score={t['score']}"
            )

        prompt = f"""You are playing a Latin America knowledge tournament match.

SITUATION:
- Topic: {ctx.topic}
- Your role this turn: {ctx.role}
- Turn: {ctx.turn}/{ctx.total_turns}
- Your accumulated score: {sum(ctx.my_scores)} pts
- Opponent score: {sum(ctx.opponent_scores)} pts
- Opponent name: {ctx.opponent_name}
{f"- Question to answer: {ctx.current_question}" if ctx.role == "answerer" else ""}

OPPONENT PROFILE:
{opp_summary}

RECENT MATCH HISTORY:{history_text if history_text else " (first turn)"}

KNOWLEDGE BASE (relevant facts retrieved):
{rag_context}

YOUR PERSISTENT NOTES:
{ctx.scratchpad or "(empty)"}

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

FINAL: <final question or answer only>
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def ask(self, ctx: MatchContext) -> str:
        """Generate a strategic question. Extracts FINAL line from think()."""
        scratchpad = self.think(ctx)
        return self._extract_final(scratchpad)

    def answer(self, ctx: MatchContext) -> str:
        """Generate a RAG-augmented answer. Extracts FINAL line from think()."""
        scratchpad = self.think(ctx)
        return self._extract_final(scratchpad)

    # ── helpers ──────────────────────────────────────────────

    def _extract_final(self, scratchpad: str) -> str:
        """Extract text after FINAL: label."""
        lines = scratchpad.split("\n")
        for i, line in enumerate(lines):
            if "FINAL:" in line.upper():
                rest = line.split(":", 1)[1].strip()
                if rest:
                    return rest
                # FINAL on its own line, answer on next
                remaining = "\n".join(lines[i+1:]).strip()
                return remaining if remaining else scratchpad.split("\n")[-1]
        # Fallback: last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return scratchpad

    def _random_move(self, ctx: WorldContext) -> Position:
        dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0),(0,0)])
        return Position(
            x=max(0, min(ctx.map_width  - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )


# ── CELL 6: Run ──────────────────────────────────────────────
agent = CondorAgent()

serve_and_register(
    agent       = agent,
    arena_url   = ARENA_URL,
    port        = 5000,
    ngrok_token = NGROK_TOKEN,
)
# This cell blocks. The agent is now live and registered.
# Wait for the organizer to accept you in the admin panel.