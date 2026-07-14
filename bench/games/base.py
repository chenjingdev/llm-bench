"""Common types for multi-turn benchmark games."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Action:
    kind: str
    value: str
    raw: str
    valid: bool = True
    error: str = ""


@dataclass
class GameState:
    game: str
    version: str
    seed: int
    max_turns: int
    secret: str
    turn: int = 0
    done: bool = False
    solved: bool = False
    stop_reason: str = ""
    history: list[dict] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    private: dict = field(default_factory=dict, repr=False)


class Game(Protocol):
    id: str
    version: str
    max_turns: int

    def reset(self, seed: int) -> GameState: ...
    def render(self, state: GameState) -> str: ...
    def parse(self, text: str) -> Action: ...
    def step(self, state: GameState, action: Action) -> dict: ...
    def result(self, state: GameState) -> dict: ...


ProgressCallback = Callable[[GameState, dict], None]
