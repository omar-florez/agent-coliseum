# ============================================================
#  LATAM ARENA — Colab 02: LangChain Agent
# ============================================================
#
#  Strategy: LangChain agent with structured memory.
#    - Uses ConversationBufferMemory to track match history
#    - Uses RunnableSequence (LCEL) for CoT
#    - Optionally uses a Retriever tool for RAG
#    - No external tools during the match — fast responses
#
#  Demonstrates how LangChain abstractions map to the arena API.
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
# !pip install flask flask-cors pyngrok langchain langchain-openai \
#              sentence-transformers faiss-cpu requests -q

# ── CELL 2: Config ───────────────────────────────────────────
import os, json, random
from agent_base import Agent, MatchContext, MatchResult, WorldContext, Position
from agent_server import serve_and_register

OPENAI_API_KEY = "sk-..."
ARENA_URL      = "https://agent-coliseum.onrender.com"
NGROK_TOKEN    = "your_ngrok_token"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── CELL 3: LangChain setup ───────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

THINK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a competitive Latin America knowledge agent.
You think carefully before asking or answering.
Always structure your response with these sections:
SITUATION / OPPONENT / GOAL / DRAFT / CRITIQUE / FINAL"""),
    ("human", """{input}"""),
])

think_chain = THINK_PROMPT | llm | StrOutputParser()

# ── CELL 4: Agent implementation ─────────────────────────────

class LangChainLatAmAgent(Agent):
    """
    LangChain-powered agent.
    Uses LCEL chain for structured thinking.
    Maintains per-match conversation memory.
    """

    name        = "LangChain Puma"
    avatar      = "🐆"
    description = "Agente construido con LangChain LCEL y memoria de conversación"

    def __init__(self):
        self._match_memory = {}  # match_id → ConversationBufferMemory
        self._opponent_notes = {}

    # ── lifecycle ─────────────────────────────────────────────

    def on_arena_start(self, ctx: WorldContext) -> None:
        self._match_memory     = {}
        self._opponent_notes   = {}
        print(f"[{self.name}] Arena started with {len(ctx.agents)} agents.")

    def on_match_start(self, ctx: MatchContext) -> None:
        # Fresh memory for each match
        self._match_memory[ctx.match_id] = ConversationBufferMemory(
            return_messages=True,
            human_prefix="Turn",
            ai_prefix="Agent",
        )
        print(f"[{self.name}] Starting match {ctx.match_id} vs {ctx.opponent_name}")

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        won = result.winner_id == ctx.my_agent_id
        # Save notes about opponent
        self._opponent_notes[ctx.opponent_agent_id] = {
            "name":    ctx.opponent_name,
            "result":  "won" if won else "lost",
            "topic":   ctx.topic,
        }
        print(f"[{self.name}] Match over: {'WON' if won else 'LOST'}")

    # ── world ─────────────────────────────────────────────────

    def move(self, ctx: WorldContext) -> Position:
        """Default random walk — implement targeting logic here."""
        dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0),(0,0)])
        return Position(
            x=max(0, min(ctx.map_width  - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    # ── cognition ─────────────────────────────────────────────

    def think(self, ctx: MatchContext) -> str:
        """
        Calls LangChain LCEL chain with structured prompt.
        Uses match memory to include conversation history.
        """
        # Build history summary
        history = ""
        for t in ctx.history[-3:]:
            history += f"\nTurn {t['turn_number']}: Q={t['question'][:50]} A={t['answer'][:50]} Score={t['score']}"

        # Opponent notes
        opp = self._opponent_notes.get(ctx.opponent_agent_id, {})
        opp_text = (f"Previously faced them: {opp.get('result','unknown')} on topic {opp.get('topic','?')}"
                    if opp else "First time meeting this opponent.")

        input_text = f"""CURRENT MATCH:
Topic: {ctx.topic}
My role: {ctx.role}  |  Turn {ctx.turn}/{ctx.total_turns}
My score: {sum(ctx.my_scores)} pts  |  Opponent: {sum(ctx.opponent_scores)} pts
Opponent: {ctx.opponent_name}
{f'Question to answer: {ctx.current_question}' if ctx.role == 'answerer' else ''}

OPPONENT NOTES: {opp_text}

HISTORY:{history if history else ' (first turn)'}

MY NOTES: {ctx.scratchpad or '(none)'}

Now think through SITUATION / OPPONENT / GOAL / DRAFT / CRITIQUE / FINAL."""

        result = think_chain.invoke({"input": input_text})

        # Store in match memory
        mem = self._match_memory.get(ctx.match_id)
        if mem:
            mem.chat_memory.add_user_message(input_text[:200])
            mem.chat_memory.add_ai_message(result[:200])

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
                if rest: return rest
                remaining = "\n".join(lines[i+1:]).strip()
                return remaining if remaining else text.split("\n")[-1]
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return text


# ── CELL 5: Run ──────────────────────────────────────────────
agent = LangChainLatAmAgent()

serve_and_register(
    agent       = agent,
    arena_url   = ARENA_URL,
    port        = 5001,       # different port from other Colabs
    ngrok_token = NGROK_TOKEN,
)
