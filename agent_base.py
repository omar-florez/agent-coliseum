"""
agent_base.py  —  The ONLY file participants need to import.

Copy this file into your Colab. Implement the Agent class.
No arena dependencies, no API keys required from you here.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random


# ── Dataclasses sent from the arena to your agent ────────────────────────────

@dataclass
class Position:
    x: int
    y: int


@dataclass
class AgentInfo:
    """Info about another agent visible on the map."""
    agent_id: str
    name:     str
    avatar:   str
    score:    int
    position: Position
    status:   str   # "active" | "eliminated"


@dataclass
class MatchContext:
    """Everything your agent knows about the current match."""
    match_id:          str
    topic:             str
    turn:              int
    total_turns:       int
    role:              str          # "asker" | "answerer"
    history:           list         # list of dicts with past turns
    my_agent_id:       str
    opponent_agent_id: str
    opponent_name:     str
    my_scores:         list         # scores I earned as answerer
    opponent_scores:   list         # scores opponent earned as answerer
    scratchpad:        str = ""     # your persistent notes across turns
    current_question:  str = ""     # the question (populated when role=answerer)


@dataclass
class MatchResult:
    match_id:     str
    winner_id:    str
    loser_id:     str
    turns:        list
    final_scores: dict
    topic:        str


@dataclass
class WorldContext:
    """Snapshot of the arena map — used for movement decisions."""
    phase:        str
    my_agent_id:  str
    my_position:  Position
    my_score:     int
    agents:       list          # list of AgentInfo for all active agents
    map_width:    int
    map_height:   int


# ── The interface ─────────────────────────────────────────────────────────────

class Agent(ABC):
    """
    Implement this class to enter the LatAm Arena.

    REQUIRED: think(), ask(), answer()
    OPTIONAL: all lifecycle and world methods (have sensible defaults)

    Your agent runs in YOUR Colab with YOUR API keys.
    The arena only calls these methods over HTTP — it never sees your keys.
    """

    name:        str = "My Agent"
    avatar:      str = "🤖"
    description: str = "A LatAm knowledge agent"

    # ── Lifecycle (optional overrides) ────────────────────────────────────────

    def on_arena_start(self, ctx: WorldContext) -> None:
        """
        Called once when the tournament begins.
        Initialize your memory structures, opponent tracking, etc.
        """
        pass

    def on_match_start(self, ctx: MatchContext) -> None:
        """
        Called before each match.
        Prepare strategy: look up this opponent in your memory.
        """
        pass

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        """
        Called after each match ends.
        Update your memory: what topics did the opponent struggle with?
        """
        pass

    def on_eliminated(self) -> None:
        """Called when you are eliminated. Log your final stats."""
        print(f"[{self.name}] Eliminated from the arena.")

    # ── World (optional overrides) ────────────────────────────────────────────

    def move(self, ctx: WorldContext) -> Position:
        """
        Return your desired next tile position.
        Smart strategy: move toward weak opponents, away from strong ones.
        Default: random walk.
        """
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
        return Position(
            x=max(0, min(ctx.map_width  - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    def should_challenge(self, ctx: WorldContext, target: AgentInfo) -> bool:
        """
        Return True to challenge target when you're adjacent.
        Default: always challenge.
        Smart strategy: only challenge when you have a topic advantage.
        """
        return True

    # ── Cognition (REQUIRED) ─────────────────────────────────────────────────

    @abstractmethod
    def think(self, ctx: MatchContext) -> str:
        """
        Your private chain-of-thought scratchpad.
        Called internally before ask() and answer().

        Structure your output as:
          SITUATION: <what's happening in the match>
          OPPONENT:  <what you know about them>
          GOAL:      <your goal this turn>
          DRAFT:     <initial response>
          CRITIQUE:  <is it good enough? what's missing?>
          FINAL:     <revised final response>

        Return the full reasoning text.
        The arena exposes a summary to the visualizer (great for the talk!).
        """
        pass

    @abstractmethod
    def ask(self, ctx: MatchContext) -> str:
        """
        Generate a strategic question on ctx.topic.
        Call self.think(ctx) first, then extract the FINAL line.
        Return only the question text.
        """
        pass

    @abstractmethod
    def answer(self, ctx: MatchContext) -> str:
        """
        Answer ctx.current_question on ctx.topic.
        Call self.think(ctx) first, then extract the FINAL line.
        Return only the answer text.
        """
        pass
