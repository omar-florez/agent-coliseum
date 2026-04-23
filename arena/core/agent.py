from abc import ABC, abstractmethod
import random
from .models import MatchContext, MatchResult, WorldContext, Position, AgentInfo


class Agent(ABC):
    """
    Base class every arena participant must implement.

    Only ask(), answer(), and think() are mandatory.
    All lifecycle and world methods have working defaults.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    name:        str = "Unnamed Agent"
    avatar:      str = "🤖"
    description: str = "An arena participant"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_arena_start(self, ctx: WorldContext) -> None:
        """Tournament begins. Initialize memory structures here."""
        pass

    def on_match_start(self, ctx: MatchContext) -> None:
        """Called before each match. Prepare strategy for this opponent."""
        pass

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        """Called after each match. Update memory with results."""
        pass

    def on_eliminated(self) -> None:
        """Called when this agent is eliminated. Log, cleanup, say goodbye."""
        pass

    # ── World ─────────────────────────────────────────────────────────────────

    def move(self, ctx: WorldContext) -> Position:
        """
        Return desired next tile position.
        Default: random walk within map bounds.
        Smart agents move toward weak opponents or away from strong ones.
        """
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
        return Position(
            x=max(0, min(ctx.map_width  - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    def should_challenge(self, ctx: WorldContext, target: AgentInfo) -> bool:
        """
        Return True to initiate a match with target agent.
        Default: always challenge.
        Smart agents consider target score, cooldown, past results.
        """
        return True

    # ── Cognition ─────────────────────────────────────────────────────────────

    @abstractmethod
    def think(self, ctx: MatchContext) -> str:
        """
        Private chain-of-thought scratchpad.
        Called internally before ask() and answer().

        Must implement:
          1. Situation analysis
          2. Opponent modeling (from history)
          3. Goal for this turn
          4. Draft response
          5. Self-critique and revision
          6. Final response (last line, extracted by ask/answer)

        Return the full reasoning trace as a string.
        The arena never sends this to the opponent.
        Expose a summary in ask()/answer() for the visualizer.
        """
        pass

    @abstractmethod
    def ask(self, ctx: MatchContext) -> str:
        """
        Generate a strategic question on ctx.topic.
        Must call self.think(ctx) internally.
        Return only the question text.
        """
        pass

    @abstractmethod
    def answer(self, ctx: MatchContext) -> str:
        """
        Answer ctx.current_question on ctx.topic.
        Must call self.think(ctx) internally.
        Return only the answer text.
        """
        pass
