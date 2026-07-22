"""에피소드 인스턴스 태그(nonce) 회귀 테스트.

배경: 같은 프롬프트가 전 레인에 바이트 동일하게 들어가면 공급자 배치 효과로 런 단위
출력 상관이 생겼다(실측: 1차 런 24레인 첫 추측이 특정 답으로 쏠림). 대책으로 에피소드마다
다른 태그를 프롬프트 맨 앞에 박아 프롬프트를 비동일화한다.

검증 포인트(브리프):
  1. 같은 에피소드 내 턴 1·2의 태그가 동일(프리픽스 캐시 유지 — 매 턴 갱신 아님).
  2. 다른 에피소드는 태그가 상이(상관 제거).
  3. episode_end 이벤트에 태그가 기록된다.
  4. 게임 버전 범프.
추가: nonce 발급기가 빠른 연속/동시 호출에서도 유일(32레인 동시 시작 충돌 방지 —
time_ns 단독은 해상도가 µs급이라 부족).

주의: semantle(v2.0.0)은 태그 줄 대신 JSON 프로토콜의 `time` 정수 필드로 대체됐다.
태그 줄 검증은 maze/rulelab/minefield 3게임에만, semantle은 별도 JSON time 검증을 둔다.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from bench import arena, client, embed
from bench.games.base import instance_tag_line, new_nonce, new_time_ns
from bench.games.maze import KoreanMaze
from bench.games.minefield import KoreanMinefield
from bench.games.rulelab import RuleLab
from bench.games.semantle import REFERENCE_WORDS, KoreanSemantle, SimilarityFeedback

_TAG_PREFIX = "게임 인스턴스 태그: "


class _StubOracle:
    """ollama 없이 결정론 순위를 주는 스텁(semantle/minefield reset 구동용).

    해시 좌표 거리로 참고 어휘 대비 순위를 매긴다 — minefield의 지뢰 추첨(순위 임계)이
    다양한 순위를 필요로 하므로 FakeOracle(4단어)가 아니라 REFERENCE_WORDS 전체를 쓴다.
    """

    words = REFERENCE_WORDS
    model = "stub-embedder"

    @property
    def metadata(self):
        return {"embedding_model": self.model, "reference_words": len(self.words),
                "vocab_digest": "sha256:stub"}

    @staticmethod
    def _coord(word: str) -> int:
        return int(hashlib.sha256(word.encode()).hexdigest()[:12], 16)

    def _sim(self, ref: str, w: str) -> float:
        return 1.0 / (1.0 + abs(self._coord(ref) - self._coord(w)))   # O(1), 가까울수록 큼

    def prepare(self, target):
        # scores 제공 → semantle 오프닝 추첨(_pick_opening_word)이 bisect 경로(빠름)를 탄다.
        return {"target": target, "scores": [self._sim(target, w) for w in self.words]}

    def evaluate(self, prepared, guess):
        ref = prepared["target"]
        if guess == ref:
            return SimilarityFeedback(1.0, 1)
        s = self._sim(ref, guess)
        rank = 1 + sum(1 for v in prepared["scores"] if v > s)
        return SimilarityFeedback(s, rank)

    def pair_cosine(self, a, b):
        if a == b:
            return 1.0
        sa, sb = set(a), set(b)
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0


# (게임 팩토리, 유효 액션 한 개) — 태그 줄을 쓰는 3게임(semantle은 JSON time으로 대체).
_TAG_GAMES = [
    ("ko-maze", lambda: KoreanMaze(), "MOVE 북"),
    ("ko-rulelab", lambda: RuleLab(), "TEST 3 4"),
    ("ko-minefield", lambda: KoreanMinefield(_StubOracle()), "GUESS 폭탄"),
]

# nonce(태그값) 유일성은 semantle 포함 4게임 공통(semantle nonce = str(time_ns)).
_ALL_FACTORIES = [
    ("ko-semantle", lambda: KoreanSemantle(_StubOracle())),
    ("ko-maze", lambda: KoreanMaze()),
    ("ko-rulelab", lambda: RuleLab()),
    ("ko-minefield", lambda: KoreanMinefield(_StubOracle())),
]


@pytest.mark.parametrize("gid,make,action", _TAG_GAMES)
def test_tag_at_top_and_constant_across_turns(gid, make, action):
    """태그 줄 게임: render 최상단에 태그 1줄, 턴이 지나도 태그 불변(프리픽스 캐시 유지)."""
    game = make()
    state = game.reset(1)
    assert state.nonce                                  # reset에서 1회 발급됨
    p0 = game.render(state)
    tag0 = p0.split("\n", 1)[0]
    assert tag0.startswith(_TAG_PREFIX)                 # 최상단 1줄
    assert state.nonce in tag0                          # 실제 태그값이 실려 있음
    assert "무시하세요" in tag0

    game.step(state, game.parse(action))                # 한 턴 진행
    p1 = game.render(state)
    tag1 = p1.split("\n", 1)[0]
    assert tag1 == tag0                                 # 턴 간 불변(매 턴 갱신 아님)
    # 태그 줄 자체가 두 프롬프트의 공통 prefix 앞부분에 그대로 있다.
    import os
    assert os.path.commonprefix([p0, p1]).startswith(tag0)


@pytest.mark.parametrize("gid,make", _ALL_FACTORIES)
def test_nonce_differs_across_episodes(gid, make):
    """같은 seed로 reset해도 에피소드마다 nonce가 다르다(레인/에피소드 비동일화)."""
    game = make()
    n1 = game.reset(1).nonce
    n2 = game.reset(1).nonce      # 동일 seed — 그래도 nonce는 달라야 한다
    n3 = game.reset(2).nonce
    assert n1 and n2 and n3
    assert len({n1, n2, n3}) == 3


def test_semantle_json_protocol_time_field():
    """semantle(v2.0.0): 태그 줄 대신 JSON `time` 정수 필드(설명·지시 없이 값만).

    time은 에피소드 내 불변(프리픽스 캐시)·에피소드 간 상이(비동일화). "무시"/"난수" 류
    문구가 프롬프트 어디에도 없어야 한다(지난 태그 실험 실패 원인 = "무시" 지시).
    """
    game = KoreanSemantle(_StubOracle())
    state = game.reset(1)
    p0 = game.render(state)
    d0 = json.loads(p0)                          # 프롬프트는 유효 JSON
    assert isinstance(d0["time"], int)
    assert str(d0["time"]) == state.nonce        # nonce와 time 통일
    for banned in ("무시", "난수", "세션", "식별", "게임 인스턴스 태그"):
        assert banned not in p0
    # 한 턴 진행 후에도 time 불변
    action = json.dumps({"발화할 단어": "폭탄"}, ensure_ascii=False)
    game.step(state, game.parse(action))
    d1 = json.loads(game.render(state))
    assert d1["time"] == d0["time"]
    # 에피소드가 바뀌면 time이 다르다
    assert json.loads(game.render(game.reset(2)))["time"] != d0["time"]


def test_reset_honors_explicit_nonce():
    """reset(seed, nonce=...)로 태그를 명시 주입 가능(하위호환 + 테스트 편의)."""
    game = KoreanMaze()
    state = game.reset(1, nonce="FIXED-TAG")
    assert state.nonce == "FIXED-TAG"
    assert game.render(state).startswith(f"{_TAG_PREFIX}FIXED-TAG — ")
    assert instance_tag_line(state) == f"{_TAG_PREFIX}FIXED-TAG — 의미 없는 난수입니다. 무시하세요.\n"


def test_semantle_reset_honors_explicit_numeric_nonce():
    """semantle: 명시 정수 nonce가 JSON time 필드로 재사용된다."""
    game = KoreanSemantle(_StubOracle())
    state = game.reset(1, nonce="1721034512345678901")
    assert state.nonce == "1721034512345678901"
    assert json.loads(game.render(state))["time"] == 1721034512345678901


def test_new_nonce_unique_under_rapid_calls():
    """time_ns 해상도가 낮아도(µs급) 원자 카운터로 유일성 보장 — 32레인 동시 충돌 방지.

    bare time_ns였다면 1000회 연속 호출에 고유값이 수십 개뿐이라 이 단언이 깨진다.
    """
    xs = [new_nonce() for _ in range(1000)]
    assert len(set(xs)) == 1000


def test_new_time_ns_monotonic_unique():
    """new_time_ns(semantle time 발급기)도 빠른 연속 호출에서 유일·단조 증가."""
    xs = [new_time_ns() for _ in range(1000)]
    assert len(set(xs)) == 1000
    assert xs == sorted(xs)


def test_all_games_version_bumped():
    """프롬프트(=측정 조건) 변경 → 게임 버전 범프(semantle은 JSON 프로토콜)."""
    assert KoreanSemantle.version == "2.0.0"
    assert KoreanMaze.version == "1.1.0"
    assert RuleLab.version == "1.1.0"
    assert KoreanMinefield.version == "1.1.0"


def test_arena_records_distinct_nonce_per_episode(monkeypatch, tmp_path):
    """엔진이 각 episode_end에 태그를 기록하고, 에피소드마다 태그가 다르다.

    verify(재생)는 프롬프트를 재구성하지 않으므로 태그와 무관하게 통과해야 한다.
    오라클 불필요 게임(rulelab)으로 ollama 없이 구동한다.
    """
    game = RuleLab(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(client, "call",
                        lambda model, prompt, **kw: client.CallResult(
                            model=model, text="TEST 1 2", cost_usd=0.0,
                            input_tokens=1, output_tokens=1, duration_ms=1, session_id="s"))

    run_dir = arena.run_arena("ko-rulelab", ["m@low"], episodes=2, max_turns=3,
                              seed_base=1, run_root=tmp_path)

    events = [json.loads(l) for l in
              (run_dir / "models" / "m@low" / "events.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    ends = [e for e in events if e["type"] == "episode_end"]
    assert len(ends) == 2
    nonces = [e.get("nonce") for e in ends]
    assert all(nonces)                        # 각 에피소드에 태그 기록
    assert nonces[0] != nonces[1]             # 에피소드마다 상이(상관 제거)
    # turn 이벤트에는 태그를 싣지 않는다(episode 단위에만 — verify가 프롬프트 미재구성).
    assert not any("nonce" in e for e in events if e["type"] == "turn")

    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["verify"]["ok"] is True        # 태그와 무관하게 재생 검증 통과
    assert man["game_version"] == "1.1.0"
