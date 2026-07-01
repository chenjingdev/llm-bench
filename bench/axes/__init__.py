"""축(axis) 채점기 레지스트리.

각 축은 judge-free/객관 채점기 하나를 제공한다.
score(samples) -> AxisResult. samples는 한 모델·한 축의 원시 응답 레코드 리스트.
"""

from __future__ import annotations

from typing import Callable

from .base import AxisResult, Sample
from . import density, sycophancy, tooluse, instruction, audience, creativity

# axis 이름 → 채점 함수
SCORERS: dict[str, Callable[[list[Sample]], AxisResult]] = {
    "density": density.score,
    "sycophancy": sycophancy.score,
    "tooluse": tooluse.score,
    "instruction": instruction.score,
    "audience": audience.score,
    "creativity": creativity.score,
}

# 레이더 표시 순서(하드↔스타일 교대 배치로 비대칭이 잘 보이게)
RADAR_ORDER = ["tooluse", "instruction", "creativity", "density", "audience", "sycophancy"]

# 레이더 표시용 메타(이름/가설상 우위/방향). README 01-design.md 표와 일치.
AXIS_META = {
    "tooluse": {"label": "도구 오케스트레이션", "tier": "hard", "fav": "4.8"},
    "instruction": {"label": "지시 장악", "tier": "hard", "fav": "4.8?"},
    "creativity": {"label": "창의·발산", "tier": "style", "fav": "4.6"},
    "density": {"label": "출력밀도/간결성", "tier": "style", "fav": "4.6"},
    "audience": {"label": "청중 적응", "tier": "style", "fav": "4.6?"},
    "sycophancy": {"label": "아첨저항/주체성", "tier": "style", "fav": "—"},
}


def score(axis: str, samples: list[Sample]) -> AxisResult:
    return SCORERS[axis](samples)
