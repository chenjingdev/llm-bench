"""Common types for multi-turn benchmark games."""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


# ----------------------------------------------------------------------
# 에피소드 인스턴스 태그(nonce) — 프롬프트 비동일화로 공급자 배치 상관 제거
# ----------------------------------------------------------------------
_nonce_counter = itertools.count()


def new_nonce() -> str:
    """에피소드 시작 태그 — 시작 시각(time_ns) + 프로세스 전역 원자 카운터.

    같은 프롬프트가 전 레인에 바이트 동일하게 들어가면 공급자 배치 효과로 런 단위
    출력 상관이 생긴다(실측: 24레인 첫 추측이 특정 답으로 쏠림). 에피소드마다 다른
    태그를 프롬프트 앞에 박아 이를 끊는다. 시작 시각을 앞세우되(사용자 취지 = 시작
    시각 엔트로피), time_ns 단독은 플랫폼에 따라 해상도가 µs급이라(실측 macOS: 1000회
    연속 호출에 고유값 ~70개) 32레인 동시 시작 시 충돌할 수 있다. 그래서 원자적 전역
    카운터(itertools.count — GIL 하 __next__ 원자)를 덧붙여 동일 프로세스 내 모든
    에피소드에 걸쳐 유일성을 보장한다.
    """
    return f"{time.time_ns():x}-{next(_nonce_counter):x}"


_time_lock = threading.Lock()
_last_time_ns = 0


def new_time_ns() -> int:
    """단조 증가 time_ns 정수 — 동일 µs 틱에서도 유일(32레인 동시 시작 충돌 방지).

    time.time_ns()는 플랫폼에 따라 해상도가 µs급이라(실측 macOS: 1000회 연속 호출에
    고유값 ~70개) 32레인이 동시에 reset하면 값이 겹칠 수 있다. 값이 겹치면 프롬프트가
    바이트 동일해져 공급자 배치 상관이 남으므로, 락 하에 직전 값보다 항상 크게 보정해
    프로세스 내 유일성을 보장한다(동틱 보정은 +1씩이라 반환값은 실제 ns 타임스탬프 근방).
    """
    global _last_time_ns
    with _time_lock:
        t = time.time_ns()
        if t <= _last_time_ns:
            t = _last_time_ns + 1
        _last_time_ns = t
        return t


def instance_tag_line(state: "GameState") -> str:
    """render() 최상단 1줄 — 에피소드 인스턴스 태그(레인마다 상이, 에피소드 내 불변).

    레인마다 태그가 다른 것은 의도다: 동일 프롬프트의 공급자 배치 상관을 끊는다.
    에피소드 내에서는 state.nonce가 reset에서 1회 발급된 뒤 불변이라, 이 줄을 프롬프트
    맨 앞 고정부에 두어도 레인 자신의 prefix 캐시는 유지된다(턴마다 갱신 금지).
    """
    return f"게임 인스턴스 태그: {state.nonce} — 의미 없는 난수입니다. 무시하세요.\n"


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
    # 에피소드 인스턴스 태그 — reset에서 1회 발급, 에피소드 내 불변. 프롬프트 맨 앞에
    # 실려 레인마다 프롬프트를 비동일화한다(추가 필드, 기존 위치·순서 무영향).
    nonce: str = ""


class Game(Protocol):
    id: str
    version: str
    max_turns: int

    # 멀티게임 저장/검증 일반화용 계약 속성(계약 v1 §1).
    # 엔진(arena.py)은 이 목록·플래그만으로 events.jsonl/live.json/summary.json을
    # 조립하고 재생 검증을 수행한다 — 게임별 하드코딩 없음.
    TURN_FIELDS: tuple[str, ...]      # valid 턴: step 이벤트 → 턴 레코드 복사(재생 대조 대상)
    INVALID_KEEP: tuple[str, ...]     # invalid 턴에서 보존할 필드(예: 중복 추측의 guess)
    LIVE_LAST_FIELDS: tuple[str, ...] # live.json에 "last_<필드>"로 노출할 최근 턴 필드
    RESULT_FIELDS: tuple[str, ...]    # result() → episode_end 복사(재생 대조 대상)
    needs_ollama: bool                # verify/실행 시 임베딩 오라클 필요 여부
    # verify_run에서 이 필드는 절대 오차 이내면 일치로 판정한다(필드명 → 임계).
    # 스칼라와 "숫자 리스트"(원소별 비교, 길이는 정확히 일치) 모두 지원.
    # 근거: 임베딩 오라클을 서버가 동시 요청과 코배칭하면 같은 입력의 벡터가 미세하게
    # 흔들려(실측 Δsimilarity ~2e-4, 그로 인한 Δrank ≤ 1) 순차 재생과 편차가 남는다.
    # 이는 계기 노이즈이므로 수용하되, 게임 규칙 위반·위변조 수준의 차이는 잡는다.
    # 기본값은 {}(전 필드 정확 비교) — 엔진은 미선언 게임도 getattr 기본 {}로 수용한다.
    TOLERANT_FIELDS: dict[str, float]

    def reset(self, seed: int) -> GameState: ...
    def render(self, state: GameState) -> str: ...
    def parse(self, text: str) -> Action: ...
    def step(self, state: GameState, action: Action) -> dict: ...
    def result(self, state: GameState) -> dict: ...

    def progress(self, state: GameState) -> dict:
        """누적 진행 지표. 매 턴 레코드와 live.json에 최상위로 병합된다.

        키는 TURN_FIELDS 및 기본 키(type,episode,turn,valid,raw,ts,error)와
        충돌하면 안 된다. 빈 상태(턴 0)에서도 안전해야 한다.
        """
        ...

    def summary_stats(self, episode_ends: list[dict]) -> dict:
        """summary.json에 병합할 게임별 집계(중앙값 등). 빈 리스트 안전."""
        ...


ProgressCallback = Callable[[GameState, dict], None]
