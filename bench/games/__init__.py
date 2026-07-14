"""ko-semantle 단일 게임 레지스트리.

이번 범위는 꼬맨틀(ko-semantle) 하나뿐이다(R1). suite/20q 구조는 계승하지 않는다.
"""

from __future__ import annotations

from .base import Action, GameState
from .semantle import KoreanSemantle


def game_names() -> list[str]:
    return ["ko-semantle"]


def build_game(name: str, *, max_turns: int | None = None):
    if name == "ko-semantle":
        return KoreanSemantle(
            max_turns=max_turns if max_turns is not None
            else KoreanSemantle.DEFAULT_MAX_TURNS)
    raise ValueError(f"unknown game: {name}")


__all__ = ["Action", "GameState", "KoreanSemantle", "build_game", "game_names"]
