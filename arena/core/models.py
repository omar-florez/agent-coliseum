from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ArenaPhase(str, Enum):
    LOBBY   = "lobby"
    ROAMING = "roaming"
    FINALS  = "finals"
    ENDED   = "ended"


class AgentStatus(str, Enum):
    PENDING    = "pending"
    ACTIVE     = "active"
    ELIMINATED = "eliminated"


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Turn:
    turn_number:         int
    asker:               str
    answerer:            str
    question:            str
    answer:              str
    score:               int
    score_reason:        str
    scratchpad_asker:    str = ""
    scratchpad_answerer: str = ""


@dataclass
class AgentInfo:
    agent_id:      str
    name:          str
    avatar:        str
    description:   str
    endpoint:      str
    status:        AgentStatus
    position:      Position
    score:         int   = 0
    wins:          int   = 0
    losses:        int   = 0
    registered_at: float = field(default_factory=time.time)


@dataclass
class MatchContext:
    match_id:          str
    topic:             str
    turn:              int
    total_turns:       int
    role:              str
    history:           list
    my_agent_id:       str
    opponent_agent_id: str
    opponent_name:     str
    my_scores:         list[int]
    opponent_scores:   list[int]
    scratchpad:        str = ""
    current_question:  str = ""


@dataclass
class MatchResult:
    match_id:     str
    winner_id:    str
    loser_id:     str
    turns:        list[Turn]
    final_scores: dict
    topic:        str


@dataclass
class WorldContext:
    phase:        ArenaPhase
    my_agent_id:  str
    my_position:  Position
    my_score:     int
    agents:       list[AgentInfo]
    map_width:    int
    map_height:   int


@dataclass
class ArenaConfig:
    max_agents:               int   = 8
    max_simultaneous_matches: int   = 3
    map_width:                int   = 20
    map_height:               int   = 15
    turns_per_match:          int   = 3
    cooldown_seconds:         int   = 30
    admin_token:              str   = "changeme"
    # Azure OpenAI (takes priority if endpoint + key are set)
    azure_openai_endpoint:    str   = ""
    azure_openai_key:         str   = ""
    azure_openai_deployment:  str   = "gpt-4o"
    # OpenAI or Azure Foundry (base_url overrides default OpenAI endpoint)
    openai_key:               str   = ""
    openai_model:             str   = "gpt-4o-mini"
    openai_base_url:          str   = ""   # e.g. https://rsgd15-foundry.openai.azure.com/openai/v1/