"""ko-rulelab(비밀 규칙 연구소) 유닛 테스트 — 오라클/임베딩/네트워크 없이 돈다.

결정론(같은 seed=같은 규칙·퀴즈), 재생 재현성(raw 재적용=동일 이벤트), TEST/ANSWER/무효
각 경로, ANSWER 즉시 종료·단일 기회, score/duplicate_tests, render 규칙·정답 미노출,
프롬프트 캐시 정렬을 검증한다.
"""

import os

from bench.games import rulelab as R
from bench.games.rulelab import RuleLab


# ----------------------------------------------------------------------
# 프로토콜 계약 속성
# ----------------------------------------------------------------------
def test_class_contract_attributes():
    assert RuleLab.id == "ko-rulelab"
    assert RuleLab.version == "1.1.0"
    assert RuleLab.DEFAULT_MAX_TURNS == 15
    assert RuleLab.needs_ollama is False
    assert RuleLab.TURN_FIELDS == ("kind", "input", "output", "answer", "correct")
    assert RuleLab.INVALID_KEEP == ()
    assert RuleLab.LIVE_LAST_FIELDS == ("kind", "input", "output")
    assert RuleLab.RESULT_FIELDS == (
        "solved", "turns", "score", "experiments", "duplicate_tests", "correct")


def test_metadata_deterministic_and_no_oracle():
    game = RuleLab()
    assert game.metadata == game.metadata
    md = game.metadata
    assert md["game"] == "ko-rulelab"
    assert md["version"] == "1.1.0"
    assert md["templates"] == len(R.RULE_TEMPLATES)


# ----------------------------------------------------------------------
# 규칙 템플릿 풀 · 출력 상한 · 결정론
# ----------------------------------------------------------------------
def test_template_pool_size_and_output_bounded():
    assert len(R.RULE_TEMPLATES) >= 12
    game = RuleLab()
    for seed in range(120):
        state = game.reset(seed)
        fn = state.private["fn"]
        for a in range(0, 21):
            for b in range(0, 21):
                out = fn(a, b)
                assert isinstance(out, int)          # 순수 정수 출력
                assert abs(out) <= 1_000_000         # |출력| ≤ 1,000,000 보장
        # 퀴즈: 서로 다른 5쌍, 도메인 안, 사전 계산된 정답과 일치
        quiz = state.private["quiz"]
        assert len(quiz) == 5 and len(set(quiz)) == 5
        for a, b in quiz:
            assert 0 <= a <= 20 and 0 <= b <= 20
        assert state.private["quiz_answers"] == [fn(a, b) for a, b in quiz]


def test_seed_determinism_same_rule_and_quiz():
    game = RuleLab()
    for seed in [0, 1, 7, 42, 123, 999]:
        s1 = game.reset(seed)
        s2 = game.reset(seed)
        assert s1.secret == s2.secret                       # 같은 규칙 설명
        assert s1.private["quiz"] == s2.private["quiz"]      # 같은 퀴즈 입력
        assert s1.private["quiz_answers"] == s2.private["quiz_answers"]
    # 시드가 바뀌면 규칙 분포가 실제로 갈린다(전부 동일 규칙 회귀 방지).
    secrets = {game.reset(s).secret for s in range(40)}
    assert len(secrets) > 3
    assert all(sec for sec in secrets)                      # 전부 비어있지 않음


# ----------------------------------------------------------------------
# 재생 재현성 — 동일 seed reset에 동일 raw 시퀀스 재적용 = 동일 이벤트
# ----------------------------------------------------------------------
def _run(seed, raws, max_turns=15):
    game = RuleLab(max_turns=max_turns)
    state = game.reset(seed)
    events = []
    for raw in raws:
        events.append(game.step(state, game.parse(raw)))
        if state.done:
            break
    return events, state


def test_replay_reproduces_events_exactly():
    raws = [
        "먼저 생각해보자.\nTEST 3 4",
        "TEST 10 2",
        "TEST 3 4",                       # 중복 실험(유효, 낭비 측정)
        "헛소리만 있는 응답",              # 무효
        "결론.\nANSWER 1 2 3 4 5",        # 제출 즉시 종료
    ]
    ev_a, st_a = _run(7, raws)
    ev_b, st_b = _run(7, raws)
    assert ev_a == ev_b                    # 재생 대조 대상: 완전 동일
    # ANSWER에서 종료 → 뒤 raw는 적용되지 않는다(단일 기회).
    assert st_a.done and st_a.stop_reason == "answered"
    assert len(ev_a) == 5


def test_replay_is_deterministic_across_many_seeds():
    raws = ["TEST 0 20", "TEST 20 0", "TEST 7 7", "ANSWER -1 -2 -3 -4 -5"]
    for seed in range(30):
        assert _run(seed, raws)[0] == _run(seed, raws)[0]


# ----------------------------------------------------------------------
# TEST 경로
# ----------------------------------------------------------------------
def test_test_action_feedback_is_rule_output():
    game = RuleLab(max_turns=15)
    state = game.reset(3)
    fn = state.private["fn"]
    ev = game.step(state, game.parse("TEST 5 6"))
    assert ev["valid"] is True
    assert ev["kind"] == "test"
    assert ev["input"] == [5, 6]
    assert ev["output"] == fn(5, 6)
    assert isinstance(ev["output"], int)
    assert state.done is False              # TEST는 종료하지 않음
    assert state.turn == 1


# ----------------------------------------------------------------------
# ANSWER 경로 — 즉시 종료 · 단일 기회 · 채점
# ----------------------------------------------------------------------
def _quiz_answer_line(state, mutate=None):
    """퀴즈 정답을 실험 없이 계산해 ANSWER 줄을 만든다(선택적으로 일부 오답 주입)."""
    answers = list(state.private["quiz_answers"])
    if mutate:
        for i in mutate:
            answers[i] += 1
    return "ANSWER " + " ".join(str(v) for v in answers)


def test_answer_all_correct_solves_and_ends():
    game = RuleLab(max_turns=15)
    state = game.reset(5)
    ev = game.step(state, game.parse(_quiz_answer_line(state)))
    assert ev["kind"] == "answer"
    assert ev["correct"] == 5
    assert ev["answer"] == state.private["quiz_answers"]
    assert state.done is True
    assert state.solved is True
    assert state.stop_reason == "answered"
    res = game.result(state)
    assert res["solved"] is True
    assert res["score"] == 1.0
    assert res["correct"] == 5


def test_answer_is_single_chance():
    # ANSWER가 done을 세우므로 뒤이은 어떤 행동도 적용되지 않는다(런 루프가 멈춘다).
    game = RuleLab(max_turns=15)
    state = game.reset(5)
    game.step(state, game.parse(_quiz_answer_line(state, mutate=[0])))
    assert state.done is True
    # 재생 루프 모사: done이면 더 이상 step하지 않는다.
    turns_before = state.turn
    assert turns_before == 1


def test_answer_partial_score_and_correct_count():
    game = RuleLab(max_turns=15)
    state = game.reset(11)
    ev = game.step(state, game.parse(_quiz_answer_line(state, mutate=[0, 1])))
    assert ev["correct"] == 3                       # 5개 중 2개 오답 주입
    res = game.result(state)
    assert res["correct"] == 3
    assert res["score"] == round(3 / 5, 6)
    assert res["solved"] is False


# ----------------------------------------------------------------------
# 무효 경로 — 원인별 한국어 오류
# ----------------------------------------------------------------------
def test_parse_invalid_paths_and_distinct_errors():
    game = RuleLab(max_turns=15)
    no_action = game.parse("아무 행동도 없는 응답")
    two_actions = game.parse("TEST 3 4\nANSWER 1 2 3 4 5")
    test_short = game.parse("TEST 3")
    test_long = game.parse("TEST 3 4 5")
    test_domain_hi = game.parse("TEST 21 4")
    test_domain_lo = game.parse("TEST -1 4")
    test_nonint = game.parse("TEST a b")
    answer_short = game.parse("ANSWER 1 2 3")
    for act in (no_action, two_actions, test_short, test_long,
                test_domain_hi, test_domain_lo, test_nonint, answer_short):
        assert act.valid is False
        assert act.error                            # 비어있지 않은 한국어 원인
    # 원인이 실제로 구분된다
    assert no_action.error != test_domain_hi.error
    assert test_short.error != test_domain_hi.error
    assert "한 개의 행동" in no_action.error
    assert "0부터 20" in test_domain_hi.error
    # 유효 경계값
    assert game.parse("TEST 0 20").valid is True
    assert game.parse("TEST 20 0").valid is True
    assert game.parse("ANSWER -5 0 100 -3 7").valid is True   # 예측은 음수 허용


def test_step_invalid_records_error_without_ending():
    game = RuleLab(max_turns=15)
    state = game.reset(1)
    ev = game.step(state, game.parse("규칙이 뭘까 고민만 함"))
    assert ev["valid"] is False
    assert ev["error"]
    assert "kind" not in ev and "output" not in ev
    assert state.turn == 1
    assert state.done is False


# ----------------------------------------------------------------------
# duplicate_tests · experiments · max_turns 소진
# ----------------------------------------------------------------------
def test_duplicate_tests_and_experiments_count():
    game = RuleLab(max_turns=15)
    state = game.reset(2)
    for raw in ["TEST 3 4", "TEST 3 4", "TEST 5 6", "TEST 3 4"]:
        game.step(state, game.parse(raw))
    res = game.result(state)
    assert res["experiments"] == 4
    assert res["duplicate_tests"] == 2               # (3,4) 3회 → 재실험 2회


def test_max_turns_without_answer_is_unsolved():
    game = RuleLab(max_turns=3)
    state = game.reset(1)
    for _ in range(3):
        game.step(state, game.parse("TEST 1 2"))
    assert state.done is True
    assert state.stop_reason == "max_turns"
    assert state.solved is False
    res = game.result(state)
    assert res["score"] == 0.0
    assert res["correct"] is None                    # 미답변 판별용
    assert res["solved"] is False
    assert res["experiments"] == 3


# ----------------------------------------------------------------------
# progress · summary_stats
# ----------------------------------------------------------------------
def test_progress_tracks_experiments_and_answered():
    game = RuleLab(max_turns=15)
    state = game.reset(1)
    assert game.progress(state) == {"experiments": 0, "answered": False}
    game.step(state, game.parse("TEST 1 2"))
    game.step(state, game.parse("헛소리"))            # 무효는 experiments에 미포함
    game.step(state, game.parse("TEST 4 5"))
    assert game.progress(state) == {"experiments": 2, "answered": False}
    game.step(state, game.parse("ANSWER 1 2 3 4 5"))
    prog = game.progress(state)
    assert prog["experiments"] == 2
    assert prog["answered"] is True


def test_summary_stats_median_and_answer_rate():
    game = RuleLab()
    assert game.summary_stats([]) == {"median_experiments": None,
                                      "answer_rate": None}
    ends = [
        {"experiments": 4, "correct": 5},
        {"experiments": 2, "correct": None},          # 미답변
        {"experiments": 6, "correct": 3},
    ]
    stats = game.summary_stats(ends)
    assert stats["median_experiments"] == 4
    assert stats["answer_rate"] == round(2 / 3, 6)


# ----------------------------------------------------------------------
# render — 규칙 설명·정답 미노출
# ----------------------------------------------------------------------
def test_render_never_reveals_rule_description():
    game = RuleLab(max_turns=15)
    for seed in range(60):
        state = game.reset(seed)
        assert state.secret                          # 규칙 설명 존재
        r0 = game.render(state)
        assert state.secret not in r0                # 빈 기록에도 미노출
        game.step(state, game.parse("TEST 3 4"))
        game.step(state, game.parse("TEST 10 15"))
        r1 = game.render(state)
        assert state.secret not in r1                # 실험 후에도 미노출
        # 퀴즈 입력은 노출되지만 퀴즈 정답은 render에 없어야 한다.
        assert "퀴즈 입력" in r1


def test_render_shows_history_lines_and_rule_block():
    game = RuleLab(max_turns=15)
    state = game.reset(3)
    fn = state.private["fn"]
    game.step(state, game.parse("TEST 4 7"))
    game.step(state, game.parse("엉뚱한 소리"))
    text = game.render(state)
    assert f"1. 실험 4 7 → {fn(4, 7)}" in text        # 계약의 기록 줄 형식
    assert "2. 형식 오류 —" in text
    assert "TEST <a> <b>" in text
    assert "ANSWER <v1> <v2> <v3> <v4> <v5>" in text
    assert "단 한 번" in text                          # ANSWER 경고


# ----------------------------------------------------------------------
# 캐시 정렬 — [고정 규칙+퀴즈] → [기록 prefix 연장] → [변동부]
# ----------------------------------------------------------------------
def test_render_prefix_stable_for_cache_alignment():
    game = RuleLab(max_turns=15)
    state = game.reset(1)
    game.step(state, game.parse("TEST 3 4"))          # 기록 1행
    p_k = game.render(state)
    game.step(state, game.parse("TEST 5 6"))          # 기록 2행(append-only 연장)
    p_k1 = game.render(state)

    common = os.path.commonprefix([p_k, p_k1])
    # 고정 규칙 블록 + 퀴즈 + 이전 기록(1행)은 공통 prefix에 들어간다.
    assert "매 응답에는 정확히 한 개의 행동만 담으세요." in common
    assert "퀴즈 입력" in common
    assert "이전 기록:" in common
    assert "1. 실험 3 4" in common                     # 기존 기록행이 prefix에 그대로
    # 변동부(현재 턴/실험 수/행동 지시)는 공통 prefix에 없다(divergence 뒤에만).
    assert "현재 턴:" not in common
    assert "지금까지 실험:" not in common
    assert "현재 턴:" in p_k[len(common):]
    assert "현재 턴:" in p_k1[len(common):]
    # 순서 확인: '이전 기록:'이 '현재 턴:'보다 앞
    assert p_k.index("이전 기록:") < p_k.index("현재 턴:")


def test_module_imports_standalone():
    # 레지스트리 미등록 상태에서도 base만 의존해 단독 import 가능해야 한다.
    import importlib
    mod = importlib.import_module("bench.games.rulelab")
    assert hasattr(mod, "RuleLab")
    assert hasattr(mod, "RULE_TEMPLATES")
