"""축 채점기 공통 타입."""

from __future__ import annotations

from dataclasses import dataclass, field

# 한 모델·한 probe의 원시 응답 레코드(runner가 JSONL로 저장한 dict)
Sample = dict


@dataclass
class AxisResult:
    axis: str
    score: float                 # 0–100 (이 축의 natural-scale 점수)
    n: int                       # 채점에 쓰인 표본 수
    subscores: dict = field(default_factory=dict)   # 세부 신호(해석용)
    detail: list = field(default_factory=list)       # probe별 디버그 정보
    note: str = ""
