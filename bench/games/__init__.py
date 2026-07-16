"""Mindmatch 게임 레지스트리.

전 게임은 games/base.py의 Game 프로토콜(계약 v1: TURN_FIELDS/RESULT_FIELDS/
progress/summary_stats 포함)을 구현한다. 여기서는 이름→클래스 매핑만 한다.
"""

from __future__ import annotations

from .base import Action, GameState
from .maze import KoreanMaze
from .minefield import KoreanMinefield
from .rulelab import RuleLab
from .semantle import KoreanSemantle

_GAMES = {
    "ko-semantle": KoreanSemantle,
    "ko-rulelab": RuleLab,
    "ko-maze": KoreanMaze,
    "ko-minefield": KoreanMinefield,
}


def game_names() -> list[str]:
    return list(_GAMES)


def build_game(name: str, *, max_turns: int | None = None):
    cls = _GAMES.get(name)
    if cls is None:
        raise ValueError(f"unknown game: {name}")
    return cls(max_turns=max_turns if max_turns is not None
               else cls.DEFAULT_MAX_TURNS)


__all__ = ["Action", "GameState", "KoreanMaze", "KoreanMinefield",
           "KoreanSemantle", "RuleLab", "build_game", "game_names"]
