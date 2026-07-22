"""ko-minefield(의미 지뢰밭) 게임 유닛 테스트.

결정론 스텁 오라클(StubOracle)을 주입해 ollama/네트워크 없이 전부 돈다. 스텁은
좌표(해시) 기반 기본 순위로 지뢰 추첨을 결정론적으로 만들고, set_rank 오버라이드로
판정 4경로(승리/폭발/경보/정상)를 정확히 제어한다. 실제 오라클 통합 테스트는
ollama 가용 시에만 도는 guard를 건다(semantle 테스트의 guard 패턴 참고).
"""

import hashlib
import math
import os
import statistics

import pytest

from bench import embed
from bench.games.minefield import (
    BOOM_RANK, LIVES, MINES, MINE_RANK_MIN, REFERENCE_WORDS, TARGET_WORDS,
    WARN_RANK, KoreanMinefield)
from bench.games.semantle import SimilarityFeedback


# ----------------------------------------------------------------------
# 결정론 스텁 오라클
# ----------------------------------------------------------------------
class StubOracle:
    """임베딩 없이 결정론 순위를 준다.

    기본 순위: 각 단어를 해시로 1-D 좌표에 놓고, ref와의 거리로 참고 어휘 대비
    순위를 매긴다(실 오라클의 '더 가까운 단어 수 + 1'과 동형). set_rank로 특정
    (기준어, 추측)의 순위를 강제해 판정 경로를 정밀 제어한다.
    """

    def __init__(self, words=REFERENCE_WORDS, model="stub-embedder"):
        self.words = tuple(dict.fromkeys(words))
        self.model = model
        self._ranks: dict[tuple[str, str], int] = {}

    @property
    def metadata(self):
        return {"type": "stub-cosine", "embedding_model": self.model,
                "reference_words": len(self.words), "vocab_digest": "sha256:stub"}

    @staticmethod
    def _coord(word: str) -> int:
        return int(hashlib.sha256(word.encode()).hexdigest()[:12], 16)

    def set_rank(self, ref: str, guess: str, rank: int) -> None:
        self._ranks[(ref, guess)] = rank

    def _rank(self, ref: str, guess: str) -> int:
        if (ref, guess) in self._ranks:
            return self._ranks[(ref, guess)]
        if guess == ref:
            return 1
        d = abs(self._coord(ref) - self._coord(guess))
        return 1 + sum(1 for w in self.words
                       if abs(self._coord(ref) - self._coord(w)) < d)

    def prepare(self, target):
        return {"target": target}

    def evaluate(self, prepared, guess):
        ref = prepared["target"]
        rank = self._rank(ref, guess)
        sim = 1.0 if guess == ref else 1.0 / (1.0 + rank)
        return SimilarityFeedback(sim, rank)

    def pair_cosine(self, a, b):
        # 결정론 가짜 코사인: 같은 단어=1.0, 아니면 글자 집합 Jaccard(semantle 테스트와 동형).
        if a == b:
            return 1.0
        sa, sb = set(a), set(b)
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0


# 정답/지뢰 단어(TARGET_WORDS·REFERENCE_WORDS)와 겹치지 않는 안전한 추측 풀.
_SAFE_GUESSES = ("폭탄", "화산", "번개", "지진", "해일", "태풍", "홍수", "가뭄")


def _game(**kw):
    return KoreanMinefield(StubOracle(), max_turns=kw.pop("max_turns", 40))


def _mines(state):
    return state.private["mine_words"]


def test_safe_guess_pool_is_disjoint():
    # 테스트가 쓰는 추측 단어는 절대 정답/지뢰가 될 수 없다(경로 제어의 전제).
    for w in _SAFE_GUESSES:
        assert w not in TARGET_WORDS
        assert w not in REFERENCE_WORDS


# ----------------------------------------------------------------------
# 결정론 + 지뢰 제약
# ----------------------------------------------------------------------
def test_same_seed_same_target_and_mines():
    game = _game()
    s1 = game.reset(7)
    s2 = game.reset(7)
    assert s1.secret == s2.secret
    assert _mines(s1) == _mines(s2)


def test_different_seeds_vary_world():
    game = _game()
    worlds = {(game.reset(seed).secret, tuple(_mines(game.reset(seed))))
              for seed in range(12)}
    assert len(worlds) > 1  # 세계가 seed에 따라 실제로 달라진다


def test_mines_satisfy_constraints():
    game = _game()
    for seed in range(25):
        state = game.reset(seed)
        prepared_target = state.private["oracle"]
        mines = _mines(state)
        assert len(mines) == MINES
        assert mines[0] != mines[1]                       # 지뢰끼리 서로 다름
        for mine in mines:
            assert mine != state.secret                   # 지뢰 ≠ 정답
            # 정답 기준 프레임에서 각 지뢰의 순위 > 30
            assert game.oracle.evaluate(prepared_target, mine).rank > MINE_RANK_MIN


def test_constants_and_protocol_attrs():
    assert (LIVES, MINES, BOOM_RANK, WARN_RANK) == (3, 2, 3, 15)
    assert KoreanMinefield.id == "ko-minefield"
    assert KoreanMinefield.version == "1.1.0"
    assert KoreanMinefield.needs_ollama is True
    assert KoreanMinefield.TURN_FIELDS == (
        "guess", "similarity", "rank", "sim_to_prev", "mine_event", "lives")
    assert KoreanMinefield.INVALID_KEEP == ("guess",)
    assert KoreanMinefield.LIVE_LAST_FIELDS == ("guess", "similarity", "rank")
    assert KoreanMinefield.RESULT_FIELDS == (
        "solved", "turns", "score", "best_rank", "best_rank_curve",
        "lives_left", "booms", "warns", "mines", "max_plateau", "fixation_sim")


def test_metadata_carries_constants_and_oracle_identity():
    game = _game()
    meta = game.metadata
    assert meta["game"] == "ko-minefield"
    assert meta["version"] == "1.1.0"
    assert (meta["lives"], meta["mines"], meta["boom_rank"], meta["warn_rank"]) == (
        3, 2, 3, 15)
    # 오라클 identity 병합(verify_run이 manifest.oracle와 대조)
    assert meta["embedding_model"] == "stub-embedder"
    assert meta["reference_words"] == len(game.oracle.words)
    assert game.metadata == game.metadata  # 결정론(재생 identity)


# ----------------------------------------------------------------------
# 판정 4경로
# ----------------------------------------------------------------------
def test_path_win_ignores_mines():
    game = _game()
    state = game.reset(3)
    event = game.step(state, game.parse(f"GUESS {state.secret}"))
    assert event["valid"] is True
    assert event["mine_event"] is None
    assert event["similarity"] == 1.0 and event["rank"] == 1
    assert state.solved is True and state.done is True
    assert state.stop_reason == "solved"
    res = game.result(state)
    assert res["solved"] is True and res["score"] > 0.5


def test_path_boom_forfeits_feedback_and_costs_life():
    game = _game()
    state = game.reset(5)
    oracle = game.oracle
    m0, m1 = _mines(state)
    boom = "폭탄"
    oracle.set_rank(m0, boom, 2)     # ≤ BOOM_RANK → 어느 한 지뢰만으로 폭발
    oracle.set_rank(m1, boom, 999)
    event = game.step(state, game.parse(f"GUESS {boom}"))
    assert event["valid"] is True
    assert event["mine_event"] == "boom"
    assert event["lives"] == LIVES - 1
    # 정보 몰수: similarity/rank/sim_to_prev 미기록
    assert "similarity" not in event
    assert "rank" not in event
    assert "sim_to_prev" not in event
    # 아직 목숨 남음 → 종료 아님
    assert state.done is False and state.stop_reason != "mined"


def test_path_warn_gives_feedback_plus_alarm():
    game = _game()
    state = game.reset(9)
    oracle = game.oracle
    m0, m1 = _mines(state)
    word = "화산"
    oracle.set_rank(state.secret, word, 40)   # 정답 피드백용 순위
    oracle.set_rank(m0, word, 10)             # 3 < 10 ≤ 15 → 경보
    oracle.set_rank(m1, word, 999)
    event = game.step(state, game.parse(f"GUESS {word}"))
    assert event["mine_event"] == "warn"
    assert event["rank"] == 40
    assert "similarity" in event and "sim_to_prev" in event
    assert event["lives"] == LIVES         # 경보는 목숨 소모 없음
    assert state.done is False


def test_path_normal_like_semantle():
    game = _game()
    state = game.reset(11)
    oracle = game.oracle
    m0, m1 = _mines(state)
    word = "번개"
    oracle.set_rank(state.secret, word, 50)
    oracle.set_rank(m0, word, 999)
    oracle.set_rank(m1, word, 999)
    event = game.step(state, game.parse(f"GUESS {word}"))
    assert event["mine_event"] is None
    assert event["rank"] == 50
    assert event["lives"] == LIVES
    assert event["similarity"] == round(1.0 / 51, 8)
    assert event["sim_to_prev"] is None    # 첫 유효 추측


def test_judgment_order_win_beats_mine():
    # 정답을 맞히면 그 정답이 지뢰 근처여도 승리(판정 순서 1이 최우선).
    game = _game()
    state = game.reset(15)
    m0, m1 = _mines(state)
    # 정답을 두 지뢰 모두에 대해 폭발권으로 강제해도 승리해야 한다.
    oracle = game.oracle
    oracle.set_rank(m0, state.secret, 1)
    oracle.set_rank(m1, state.secret, 1)
    event = game.step(state, game.parse(f"GUESS {state.secret}"))
    assert event["mine_event"] is None
    assert state.solved is True and state.stop_reason == "solved"


# ----------------------------------------------------------------------
# 목숨 소진 종료
# ----------------------------------------------------------------------
def test_lives_exhaustion_ends_game_mined():
    game = _game()
    state = game.reset(13)
    oracle = game.oracle
    m0, m1 = _mines(state)
    words = ["폭탄", "화산", "번개"]
    for w in words:
        oracle.set_rank(m0, w, 1)   # 매 추측 폭발
        oracle.set_rank(m1, w, 999)

    for i, w in enumerate(words):
        event = game.step(state, game.parse(f"GUESS {w}"))
        assert event["mine_event"] == "boom"
        assert event["lives"] == LIVES - (i + 1)

    assert state.done is True and state.stop_reason == "mined"
    res = game.result(state)
    assert res["lives_left"] == 0
    assert res["booms"] == 3
    assert res["warns"] == 0
    assert res["solved"] is False
    assert res["score"] == 0.0        # mined → 점수 0


def test_progress_reports_best_rank_and_lives():
    game = _game()
    state = game.reset(13)
    oracle = game.oracle
    m0, m1 = _mines(state)
    # 정상 추측(rank 20) → best_rank 20, lives 유지
    oracle.set_rank(state.secret, "번개", 20)
    oracle.set_rank(m0, "번개", 999)
    oracle.set_rank(m1, "번개", 999)
    game.step(state, game.parse("GUESS 번개"))
    assert game.progress(state) == {"best_rank": 20, "lives": LIVES}
    # 폭발 → lives 감소, best_rank는 폭발 턴 무관하게 유지
    oracle.set_rank(m0, "폭탄", 1)
    oracle.set_rank(m1, "폭탄", 999)
    game.step(state, game.parse("GUESS 폭탄"))
    assert game.progress(state) == {"best_rank": 20, "lives": LIVES - 1}


# ----------------------------------------------------------------------
# best_rank_curve의 폭발 턴 처리 + 폭발 이벤트 필드 부재
# ----------------------------------------------------------------------
def test_best_rank_curve_holds_prev_on_boom():
    game = _game()
    state = game.reset(17)
    oracle = game.oracle
    m0, m1 = _mines(state)
    # t1 정상 rank8 → t2 폭발 → t3 정상 rank12(개선 없음)
    oracle.set_rank(state.secret, "번개", 8)
    oracle.set_rank(m0, "번개", 999)
    oracle.set_rank(m1, "번개", 999)
    oracle.set_rank(m0, "폭탄", 1)
    oracle.set_rank(m1, "폭탄", 999)
    oracle.set_rank(state.secret, "화산", 12)
    oracle.set_rank(m0, "화산", 999)
    oracle.set_rank(m1, "화산", 999)

    game.step(state, game.parse("GUESS 번개"))
    boom = game.step(state, game.parse("GUESS 폭탄"))
    game.step(state, game.parse("GUESS 화산"))

    res = game.result(state)
    # 폭발 턴은 rank 없음 → curve에 직전 best(8) 유지, t3(rank12)도 개선 아님.
    assert res["best_rank_curve"] == [8, 8, 8]
    assert res["best_rank"] == 8
    assert res["booms"] == 1
    # 폭발 이벤트엔 순위 정보가 없다(재확인)
    assert "rank" not in boom and "similarity" not in boom


def test_boom_guess_included_in_sim_chain():
    # 폭발 턴 추측도 '직전 유효 추측'으로 다음 턴 sim_to_prev에 반영된다.
    game = _game()
    state = game.reset(17)
    oracle = game.oracle
    m0, m1 = _mines(state)
    oracle.set_rank(m0, "화산", 1)      # 폭발
    oracle.set_rank(m1, "화산", 999)
    oracle.set_rank(state.secret, "홍수", 30)   # 정상, '화산'과 sim_to_prev 계산
    oracle.set_rank(m0, "홍수", 999)
    oracle.set_rank(m1, "홍수", 999)
    game.step(state, game.parse("GUESS 화산"))   # 폭발(sim_to_prev 미기록)
    ev = game.step(state, game.parse("GUESS 홍수"))
    # prev = 폭발 추측 '화산' → Jaccard(화산, 홍수) = 공유 '' / {화,산,홍,수} = 0.0
    assert ev["sim_to_prev"] == 0.0


# ----------------------------------------------------------------------
# 고착 지표(폭발 턴 제외, semantle 정의)
# ----------------------------------------------------------------------
def test_fixation_metrics_over_ranked_turns():
    game = _game()
    state = game.reset(3)
    oracle = game.oracle
    m0, m1 = _mines(state)
    for w in ("생활", "생각", "활동"):        # 전부 정상, rank 5 고정 → 정체 2턴
        oracle.set_rank(state.secret, w, 5)
        oracle.set_rank(m0, w, 999)
        oracle.set_rank(m1, w, 999)
        game.step(state, game.parse(f"GUESS {w}"))
    res = game.result(state)
    assert res["max_plateau"] == 2
    s_t2, s_t3 = round(1 / 3, 8), 0.0     # 생활→생각 공유 '생', 생각→활동 공유 없음
    assert res["fixation_sim"] == round(statistics.median([s_t2, s_t3]), 8)


# ----------------------------------------------------------------------
# 무효 턴(형식 오류/중복) — semantle과 동일
# ----------------------------------------------------------------------
def test_invalid_and_duplicate_turns():
    game = _game()
    state = game.reset(11)
    oracle = game.oracle
    m0, m1 = _mines(state)
    oracle.set_rank(state.secret, "번개", 30)
    oracle.set_rank(m0, "번개", 999)
    oracle.set_rank(m1, "번개", 999)

    bad = game.step(state, game.parse("아무 행동 없음"))
    assert bad["valid"] is False and "guess" not in bad
    game.step(state, game.parse("GUESS 번개"))
    dup = game.step(state, game.parse("GUESS 번개"))
    assert dup["valid"] is False
    assert dup["error"] == "duplicate guess"
    assert dup["guess"] == "번개"          # INVALID_KEEP 보존


# ----------------------------------------------------------------------
# render — 정답/지뢰 미노출 + 목숨/경보/폭발 표기
# ----------------------------------------------------------------------
def test_render_hides_target_and_mines():
    game = _game()
    state = game.reset(19)
    oracle = game.oracle
    m0, m1 = _mines(state)
    oracle.set_rank(state.secret, "번개", 30)   # 정상
    oracle.set_rank(m0, "번개", 999)
    oracle.set_rank(m1, "번개", 999)
    oracle.set_rank(m0, "폭탄", 1)               # 폭발
    oracle.set_rank(m1, "폭탄", 999)
    game.step(state, game.parse("GUESS 번개"))
    game.step(state, game.parse("GUESS 폭탄"))
    text = game.render(state)

    # 동적 영역(기록·피드백·변동부)에 정답/지뢰 단어가 절대 새지 않는다.
    dynamic = text[text.index("이전 기록:"):]
    assert state.secret not in dynamic
    assert m0 not in dynamic and m1 not in dynamic

    # 지뢰 규칙·목숨 줄·폭발/정상 표기는 보인다.
    assert "지뢰" in text
    assert "남은 목숨:" in text
    assert "번개" in text                 # 모델 자신의 추측은 표시
    assert "지뢰 폭발!" in text            # 폭발 턴 표기
    assert "폭탄" in text


def test_render_warn_and_normal_lines():
    game = _game()
    state = game.reset(21)
    oracle = game.oracle
    m0, m1 = _mines(state)
    oracle.set_rank(state.secret, "화산", 40)   # 경보
    oracle.set_rank(m0, "화산", 10)
    oracle.set_rank(m1, "화산", 999)
    game.step(state, game.parse("GUESS 화산"))
    text = game.render(state)
    n = len(game.oracle.words)
    assert f"{n}개 중 40위" in text
    assert "[지뢰 접근 경보]" in text


# ----------------------------------------------------------------------
# 재생 재현성 — 동일 seed reset + 저장 raw 재적용 = 동일 이벤트
# ----------------------------------------------------------------------
def test_replay_reproduces_events():
    game = _game()
    # seed 세계를 읽어 스크립트를 무장한다(오버라이드는 오라클에 상주 → 재생에도 유효).
    probe = game.reset(21)
    oracle = game.oracle
    m0, m1 = _mines(probe)
    oracle.set_rank(probe.secret, "번개", 20)   # 정상
    oracle.set_rank(m0, "번개", 999)
    oracle.set_rank(m1, "번개", 999)
    oracle.set_rank(m0, "폭탄", 2)               # 폭발
    oracle.set_rank(m1, "폭탄", 999)
    oracle.set_rank(probe.secret, "화산", 6)     # 경보
    oracle.set_rank(m0, "화산", 10)
    oracle.set_rank(m1, "화산", 999)

    raws = ["GUESS 번개", "설명만 하고 행동 없음", "GUESS 폭탄", "GUESS 화산"]

    def play():
        state = game.reset(21)
        return [game.step(state, game.parse(r)) for r in raws]

    first = play()
    second = play()
    assert first == second                      # 결정론(재생 대조 대상)
    # 스크립트가 실제로 4경로를 밟았는지 확인
    assert first[0]["mine_event"] is None
    assert first[1]["valid"] is False
    assert first[2]["mine_event"] == "boom" and first[2]["lives"] == LIVES - 1
    assert first[3]["mine_event"] == "warn"


# ----------------------------------------------------------------------
# 캐시 정렬 — render prefix 연장성([규칙]→[기록]→[변동부])
# ----------------------------------------------------------------------
def test_render_prefix_stable_for_cache_alignment():
    game = _game()
    state = game.reset(23)
    oracle = game.oracle
    m0, m1 = _mines(state)
    for w, r in (("번개", 8), ("화산", 12)):    # 둘 다 정상(폭발 텍스트가 기록에 안 섞이게)
        oracle.set_rank(state.secret, w, r)
        oracle.set_rank(m0, w, 999)
        oracle.set_rank(m1, w, 999)

    game.step(state, game.parse("GUESS 번개"))
    p_k = game.render(state)                    # 기록 1행
    game.step(state, game.parse("GUESS 화산"))
    p_k1 = game.render(state)                   # 기록 2행(append-only)

    common = os.path.commonprefix([p_k, p_k1])
    assert "매 응답에는 정확히 한 개의 행동만 포함하세요." in common   # 고정 규칙
    assert "이전 기록:" in common
    assert "1. 번개" in common                   # 기존 기록행이 prefix에 그대로
    # 변동부는 공통 prefix에 없다
    assert "현재 턴:" not in common
    assert "남은 목숨:" not in common
    assert "지금까지 최고:" not in common
    # 각 프롬프트에서 변동부는 공통 prefix 뒤 꼬리에 존재
    assert "현재 턴:" in p_k[len(common):]
    assert "남은 목숨:" in p_k[len(common):]
    assert p_k.index("이전 기록:") < p_k.index("현재 턴:")


# ----------------------------------------------------------------------
# summary_stats — 빈 리스트 안전 + 집계 정의
# ----------------------------------------------------------------------
def test_summary_stats_empty_and_aggregate():
    game = _game()
    assert game.summary_stats([]) == {
        "median_best_rank": None, "median_booms": None, "mined_rate": 0.0}
    ends = [
        {"best_rank": 5, "booms": 0, "lives_left": 2},
        {"best_rank": 15, "booms": 3, "lives_left": 0},   # mined
        {"best_rank": None, "booms": 1, "lives_left": 1},
    ]
    stats = game.summary_stats(ends)
    assert stats["median_best_rank"] == statistics.median([5, 15])
    assert stats["median_booms"] == statistics.median([0, 3, 1])
    assert stats["mined_rate"] == round(1 / 3, 4)


# ----------------------------------------------------------------------
# 실제 오라클 통합 — ollama 가용 시에만
# ----------------------------------------------------------------------
def test_real_oracle_smoke_if_available():
    if not embed.available():
        pytest.skip("ollama 미가용 — 실 오라클 통합 테스트 건너뜀")
    game = KoreanMinefield(max_turns=40)        # 실제 EmbeddingOracle(qwen3-embedding-honcho-8192)
    state = game.reset(1)
    assert state.secret in TARGET_WORDS
    mines = _mines(state)
    assert len(mines) == MINES and mines[0] != mines[1]
    prepared_target = state.private["oracle"]
    for mine in mines:
        assert mine != state.secret
        assert game.oracle.evaluate(prepared_target, mine).rank > MINE_RANK_MIN
    # 승리 경로 스모크
    event = game.step(state, game.parse(f"GUESS {state.secret}"))
    assert event["mine_event"] is None and state.solved is True
