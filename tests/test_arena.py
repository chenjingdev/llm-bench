"""Mindmatch 엔진 테스트 — ollama·네트워크·실모델 호출 없이 돈다.

가짜 오라클(FakeOracle)을 KoreanSemantle에 주입하고, client.call·embed.available·
build_game을 monkeypatch로 대체해 저장 계약(v2: 참가자=모델×effort)과 재생 검증을 검사한다.
"""

import json
import os
import shutil
import socket
import statistics
import threading
import time

from bench import arena, client, config, embed
from bench.games import build_game, game_names
from bench.games import semantle as sm
from bench.games.semantle import KoreanSemantle, SimilarityFeedback


# ----------------------------------------------------------------------
# 가짜 오라클 — 임베딩/네트워크 없이 결정론적 순위를 준다.
# ----------------------------------------------------------------------
class FakeOracle:
    words = ("학교", "의사", "병원", "자전거")
    model = "fake-embedder"

    @property
    def metadata(self):
        return {"embedding_model": self.model, "reference_words": len(self.words),
                "vocab_digest": "sha256:fake"}

    def prepare(self, target):
        return {"target": target}

    def evaluate(self, prepared, guess):
        ranks = {prepared["target"]: 1, "의사": 2, "학교": 4}
        rank = ranks.get(guess, 3)
        return SimilarityFeedback(1.0 / rank, rank)

    def pair_cosine(self, a, b):
        # 결정론적 가짜 코사인: 같은 단어=1.0, 아니면 글자 집합 Jaccard.
        if a == b:
            return 1.0
        sa, sb = set(a), set(b)
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0


# 정답을 모르는 척하는 단순 봇: render 프롬프트의 현재 턴 번호로 단어를 고른다.
_BOT_WORDS = ["의사", "학교", "병원", "자전거", "바다", "친구"]


def _bot_call(model, prompt, **kwargs):
    import re
    m = re.search(r"현재 턴: (\d+)/", prompt)
    idx = (int(m.group(1)) - 1) if m else 0
    word = _BOT_WORDS[idx % len(_BOT_WORDS)]
    return client.CallResult(model=model, text=f"생각 중...\nGUESS {word}",
                             cost_usd=0.0, input_tokens=1, output_tokens=1,
                             duration_ms=1, session_id="s")


def _patch_engine(monkeypatch, game):
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(client, "call", _bot_call)
    monkeypatch.setattr(embed, "available", lambda: True)


# ----------------------------------------------------------------------
# 게임 계층: parse / step / score / 결정론
# ----------------------------------------------------------------------
def test_parse_requires_exactly_one_guess():
    game = KoreanSemantle(FakeOracle(), max_turns=4)
    assert game.parse("설명만 하고 GUESS 없음").valid is False
    assert game.parse("GUESS 학교\nGUESS 병원").valid is False
    ok = game.parse("먼저 생각합니다.\nGUESS 학교")
    assert ok.valid is True
    assert ok.value == "학교"


def test_step_rejects_duplicate_and_reports_schema():
    game = KoreanSemantle(FakeOracle(), max_turns=4)
    state = game.reset(1)
    ev = game.step(state, game.parse("GUESS 의사"))
    assert ev["valid"] is True
    assert ev["guess"] == "의사"
    assert ev["rank"] == 2
    assert ev["similarity"] == round(0.5, 8)
    dup = game.step(state, game.parse("GUESS 의사"))
    assert dup["valid"] is False
    assert dup["error"] == "duplicate guess"
    assert dup["guess"] == "의사"


def test_score_solved_and_unsolved():
    solved_game = KoreanSemantle(FakeOracle(), max_turns=4)
    s = solved_game.reset(11)
    solved_game.step(s, solved_game.parse(f"GUESS {s.secret}"))
    res = solved_game.result(s)
    assert res["solved"] is True
    assert res["stop_reason"] == "solved"
    assert res["score"] == 1.0

    miss_game = KoreanSemantle(FakeOracle(), max_turns=1)
    st = miss_game.reset(1)
    wrong = next(w for w in _BOT_WORDS if w != st.secret)
    miss_game.step(st, miss_game.parse(f"GUESS {wrong}"))
    r2 = miss_game.result(st)
    assert r2["solved"] is False
    assert r2["stop_reason"] == "max_turns"
    assert 0.0 <= r2["score"] < 0.5


def test_seed_determinism_same_target():
    game = KoreanSemantle(FakeOracle(), max_turns=4)
    assert game.reset(7).secret == game.reset(7).secret
    a = KoreanSemantle(FakeOracle()).reset(123).secret
    b = KoreanSemantle(FakeOracle()).reset(123).secret
    assert a == b


def test_registry_only_semantle():
    assert game_names() == ["ko-semantle"]
    assert build_game("ko-semantle").max_turns == 40
    assert build_game("ko-semantle", max_turns=12).max_turns == 12


# ----------------------------------------------------------------------
# v1.1.0: 1글자 허용 · 오류 원인 구분 · 백분위 · sim_to_prev · 고착 지표
# ----------------------------------------------------------------------
def test_version_is_1_3_0():
    assert KoreanSemantle.version == "1.3.0"


def test_oracle_model_reflected_in_metadata(monkeypatch):
    # 실제 오라클: 임베딩/네트워크 없이 metadata의 embedding_model이 선택 모델을 담는지.
    monkeypatch.setattr(sm.embed, "model_info",
                        lambda m: {"name": m, "digest": "sha256:deadbeef"})
    monkeypatch.setattr(sm.embed, "embed",
                        lambda words, prefix=True, *, model=None: [[0.0]] * len(words))
    # 아레나 기본 오라클 모델이 qwen3-embedding:8b로 이관됐다
    assert sm.ORACLE_MODEL == "qwen3-embedding:8b"
    default_oracle = sm.EmbeddingOracle(words=("가", "나"))
    assert default_oracle.metadata["embedding_model"] == "qwen3-embedding:8b"
    assert default_oracle.metadata["embedding_digest"] == "sha256:deadbeef"
    # 명시 모델도 그대로 반영
    explicit = sm.EmbeddingOracle(words=("가", "나"), model="qwen3-embedding:4b")
    assert explicit.metadata["embedding_model"] == "qwen3-embedding:4b"
    # FakeOracle 계열은 무영향(자체 metadata 유지)
    assert FakeOracle().metadata["embedding_model"] == "fake-embedder"


def test_single_char_guess_is_valid():
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    action = game.parse("GUESS 자")          # 1글자 — v1.1.0에서 허용
    assert action.valid is True
    assert action.value == "자"
    state = game.reset(1)                     # '자'는 비타깃이라 해결되지 않음
    ev = game.step(state, action)
    assert ev["valid"] is True
    assert ev["guess"] == "자"


def test_parse_error_messages_distinguish_cause():
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    none_line = game.parse("아무 행동도 없습니다")          # GUESS 줄 0개
    two_lines = game.parse("GUESS 바다\nGUESS 하늘")        # GUESS 줄 2개
    bad_word = game.parse("GUESS hello")                    # 인자가 한국어 단어 아님
    assert none_line.valid is two_lines.valid is bad_word.valid is False
    assert "한 개" in none_line.error and "한 개" in two_lines.error
    assert "한국어 단어" in bad_word.error
    assert none_line.error != bad_word.error                # 원인이 구분된다


def test_render_shows_percentile_and_rule():
    game = KoreanSemantle(FakeOracle(), max_turns=5)        # 어휘 4개
    state = game.reset(1)
    game.step(state, game.parse("GUESS 의사"))              # rank 2 / 4 → 상위 50%
    text = game.render(state)
    assert "4개 중 2위" in text
    assert "상위 50%" in text
    assert "지금까지 최고: 2위 (상위 50%)" in text
    assert "한 글자 이상의 한국어 단어" in text


def test_render_prefix_stable_for_cache_alignment():
    # 연속 두 턴의 프롬프트 공통 prefix가 (규칙 + 이전 기록 전체)를 포함하고,
    # 변동부(현재 턴/최고/출력 지시)는 그 뒤에만 와야 캐시가 정렬된다.
    game = KoreanSemantle(FakeOracle(), max_turns=10)
    state = game.reset(1)
    game.step(state, game.parse("GUESS 활동"))     # 비타깃(rank 3) → 해결 안 됨
    p_k = game.render(state)                        # 기록 1행
    game.step(state, game.parse("GUESS 생각"))
    p_k1 = game.render(state)                       # 기록 2행(append-only 연장)

    common = os.path.commonprefix([p_k, p_k1])
    assert "매 응답에는 정확히 한 개의 행동만 포함하세요." in common   # 고정 규칙
    assert "이전 기록:" in common
    assert "1. 활동" in common                       # 기존 기록행이 prefix에 그대로
    # 변동부는 공통 prefix에 없다(divergence 뒤에만)
    assert "현재 턴:" not in common
    assert "지금까지 최고:" not in common
    # 그리고 각 프롬프트에서 변동부는 공통 prefix 뒤 꼬리에 존재
    assert "현재 턴:" in p_k[len(common):]
    assert "현재 턴:" in p_k1[len(common):]
    # 규칙+기록 순서 확인: '이전 기록:'이 '현재 턴:'보다 앞
    assert p_k.index("이전 기록:") < p_k.index("현재 턴:")


def test_sim_to_prev_is_deterministic():
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    guesses = ["생활", "생각", "활동"]                       # 전부 비타깃 → 해결 안 됨

    def run():
        s = game.reset(1)
        return [game.step(s, game.parse(f"GUESS {w}")) for w in guesses]

    a, b = run(), run()
    assert a == b                                            # 결정론(재생 대조 대상)
    assert a[0]["sim_to_prev"] is None                       # 첫 유효 추측
    assert a[1]["sim_to_prev"] == round(1 / 3, 8)            # 생활→생각: 공유 '생'
    assert a[2]["sim_to_prev"] == 0.0                        # 생각→활동: 공유 없음


def test_fixation_metrics_max_plateau_and_sim():
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    state = game.reset(3)
    for w in ["생활", "생각", "활동"]:                       # 모두 rank 3(비타깃)
        game.step(state, game.parse(f"GUESS {w}"))
    res = game.result(state)
    # best_rank가 t1 이후 개선되지 않음 → 정체 2턴
    assert res["max_plateau"] == 2
    # 정체 구간(t2,t3)의 sim_to_prev 중앙값
    s_t2, s_t3 = round(1 / 3, 8), 0.0
    assert res["fixation_sim"] == round(statistics.median([s_t2, s_t3]), 8)


def test_fixation_sim_null_without_plateau():
    # 매 턴 순위가 개선되면 정체 구간이 없어 fixation_sim은 null.
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    state = game.reset(5)
    game.step(state, game.parse("GUESS 활동"))               # rank 3 (비타깃)
    game.step(state, game.parse("GUESS 의사"))               # rank 2 (개선)
    res = game.result(state)
    assert res["max_plateau"] == 0
    assert res["fixation_sim"] is None


# ----------------------------------------------------------------------
# 참가자(모델×effort) 정규화
# ----------------------------------------------------------------------
def test_participant_slug_parsing_and_dedup():
    parts = arena._normalize_participants(
        ["m@low", "m@high", "m", {"model": "m", "effort": "low"}, "codex-5.6-luna"],
        default_effort="low")
    slugs = [p["slug"] for p in parts]
    # "m"(→m@low)와 {"model":"m","effort":"low"}는 "m@low"와 중복 → 제거
    assert slugs == ["m@low", "m@high", "codex-5.6-luna@low"]
    assert parts[0] == {"model": "m", "effort": "low", "slug": "m@low"}
    assert parts[2]["model"] == "codex-5.6-luna"


def test_participant_rejects_unknown_effort():
    import pytest
    with pytest.raises(ValueError) as ei:
        arena._normalize_participants(["m@turbo"], default_effort="low")
    assert "오타" in str(ei.value)          # 존재하지 않는 단계 = 오타


def test_participant_rejects_effort_unsupported_by_model():
    import pytest
    # codex CLI엔 max가 없다 → 거부하고 모델별 허용 목록을 메시지에 담는다.
    with pytest.raises(ValueError) as ei:
        arena._normalize_participants(["codex-5.6-sol@max"], default_effort="low")
    msg = str(ei.value)
    assert "codex-5.6-sol" in msg
    assert "low, medium, high, xhigh" in msg
    assert "max" not in msg.split("허용:")[1]   # 허용 목록에 max 없음

    # agy 모델은 표시명 접미사로 지원 단계가 정해짐 — 미지원 단계는 거부.
    with pytest.raises(ValueError):   # Pro는 low/high만 → medium 미지원
        arena._normalize_participants(["gemini-3-pro@medium"], default_effort="high")
    with pytest.raises(ValueError):   # Flash는 low/medium/high → xhigh 없음
        arena._normalize_participants(["gemini-3.5-flash@xhigh"], default_effort="high")
    with pytest.raises(ValueError):   # OSS 120B는 medium만 → low 거부
        arena._normalize_participants(["gpt-oss-120b@low"], default_effort="high")
    with pytest.raises(ValueError):   # codex는 4단계 → max 없음
        arena._normalize_participants(["codex-5.6-terra@max"], default_effort="low")


def test_participant_allows_model_supported_efforts():
    # 실호출 검증된 모델별 지원 단계는 통과한다.
    cases = {
        "gemini-3-pro@low": "gemini-3-pro@low",
        "gemini-3.5-flash@medium": "gemini-3.5-flash@medium",
        "gpt-oss-120b@medium": "gpt-oss-120b@medium",
        "codex-5.6-sol@xhigh": "codex-5.6-sol@xhigh",
        "codex-5.6-terra@xhigh": "codex-5.6-terra@xhigh",
        "claude-fable-5@max": "claude-fable-5@max",     # 네이티브 클로드 → 5단계
        "claude-haiku-4-5@max": "claude-haiku-4-5@max",
    }
    for item, slug in cases.items():
        parts = arena._normalize_participants([item], default_effort="low")
        assert parts[0]["slug"] == slug


def test_gemini_display_composes_effort_suffix():
    # agy 표시명 = base + " (Effort)" — agy models 표기와 정확히 일치해야 한다.
    assert client._gemini_display("gemini-3.5-flash", "medium") == "Gemini 3.5 Flash (Medium)"
    assert client._gemini_display("gemini-3-pro", "high") == "Gemini 3.1 Pro (High)"
    assert client._gemini_display("gemini-3-pro", "low") == "Gemini 3.1 Pro (Low)"
    assert client._gemini_display("gpt-oss-120b", "medium") == "GPT-OSS 120B (Medium)"


def test_catalog_efforts_reflect_real_calls():
    assert config.model_efforts("codex-5.6-sol") == ("low", "medium", "high", "xhigh")
    assert config.model_efforts("gemini-3-pro") == ("low", "high")
    assert config.model_efforts("gemini-3.5-flash") == ("low", "medium", "high")
    assert config.model_efforts("gpt-oss-120b") == ("medium",)
    assert config.vendor("gpt-oss-120b") == "gemini"       # agy 라우팅
    assert config.CODEX_MODELS["codex-5.4-mini"] == "gpt-5.4-mini"
    assert config.CODEX_MODELS["codex-5.6-terra"] == "gpt-5.6-terra"
    assert config.alias("codex-5.6-sol") == "5.6 Sol"
    assert config.alias("codex-5.6-terra") == "5.6 Terra"
    assert config.alias("gemini-3.5-flash") == "G3.5 Flash"
    # claude-fable-5: 네이티브 클로드 경로 → vendor claude, effort 5단계, 별칭 f5
    assert config.vendor("claude-fable-5") == "claude"
    assert config.model_efforts("claude-fable-5") == ("low", "medium", "high", "xhigh", "max")
    assert config.model_efforts("codex-5.6-terra") == ("low", "medium", "high", "xhigh")
    assert config.alias("claude-fable-5") == "f5"


# ----------------------------------------------------------------------
# 엔진 계층: 저장 계약(v2) + 재생 검증
# ----------------------------------------------------------------------
_LIVE_KEYS = {"model", "effort", "episode", "turn", "max_turns", "phase",
              "last_guess", "last_similarity", "last_rank", "best_rank",
              "raw_snippet", "updated_at"}


def test_same_model_two_efforts_stored_separately(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    _patch_engine(monkeypatch, game)

    run_dir = arena.run_arena(
        "ko-semantle", ["claude-haiku-4-5@low", "claude-haiku-4-5@high"],
        episodes=2, max_turns=3, effort="low", seed_base=100, run_root=tmp_path)

    # 같은 모델의 두 effort가 별도 slug 디렉토리에 저장된다
    low = run_dir / "models" / "claude-haiku-4-5@low"
    high = run_dir / "models" / "claude-haiku-4-5@high"
    assert low.is_dir() and high.is_dir()

    for slug_dir, effort in [(low, "low"), (high, "high")]:
        live = json.loads((slug_dir / "live.json").read_text(encoding="utf-8"))
        assert set(live.keys()) == _LIVE_KEYS
        assert live["model"] == "claude-haiku-4-5"   # 순수 모델 id
        assert live["effort"] == effort
        assert live["phase"] == "done"
        assert "target" not in live

        summary = json.loads((slug_dir / "summary.json").read_text(encoding="utf-8"))
        assert set(summary) == {"model", "effort", "episodes", "mean_score",
                                "solve_rate", "median_turns", "median_max_plateau",
                                "median_fixation_sim", "invalid_actions"}
        assert summary["model"] == "claude-haiku-4-5"
        assert summary["effort"] == effort
        assert len(summary["episodes"]) == 2

    # manifest: participants + models(slug 리스트) + 기본 effort
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["participants"] == [
        {"model": "claude-haiku-4-5", "effort": "low", "slug": "claude-haiku-4-5@low"},
        {"model": "claude-haiku-4-5", "effort": "high", "slug": "claude-haiku-4-5@high"},
    ]
    assert manifest["models"] == ["claude-haiku-4-5@low", "claude-haiku-4-5@high"]
    assert manifest["effort"] == "low"
    assert manifest["verify"]["ok"] is True

    # index 최신 항목은 slug 리스트를 노출
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["runs"][0]["models"] == ["claude-haiku-4-5@low", "claude-haiku-4-5@high"]

    assert arena.verify_run(run_dir)["ok"] is True


def test_storage_contract_and_no_target_leak(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    _patch_engine(monkeypatch, game)
    run_dir = arena.run_arena("ko-semantle", ["m-a"], episodes=2, max_turns=3,
                              effort="low", seed_base=100, run_root=tmp_path)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seeds"] == [100, 101]
    slug_dir = run_dir / "models" / "m-a@low"
    events = [json.loads(l) for l in
              (slug_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    turns = [e for e in events if e["type"] == "turn"]
    ends = [e for e in events if e["type"] == "episode_end"]
    assert len(ends) == 2 and turns
    for e in turns:
        assert "target" not in e          # 정답 누출 금지
        if e["valid"]:
            assert {"guess", "similarity", "rank"} <= set(e)
    for e in ends:
        assert "target" in e              # target은 episode_end에만
        assert set(e) >= {"solved", "turns", "best_rank", "score", "best_rank_curve"}


def test_effort_is_passed_to_client(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=1)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)
    seen = []
    lock = threading.Lock()

    def rec_call(model, prompt, *, effort="low", **kwargs):
        with lock:
            seen.append((model, effort))
        return client.CallResult(model=model, text="GUESS 의사", cost_usd=0.0,
                                 input_tokens=1, output_tokens=1, duration_ms=1,
                                 session_id="s")

    monkeypatch.setattr(client, "call", rec_call)
    arena.run_arena("ko-semantle", ["m@low", "m@high"], episodes=1, max_turns=1,
                    effort="low", seed_base=1, run_root=tmp_path)
    assert ("m", "low") in seen
    assert ("m", "high") in seen


def test_workers_param_caps_concurrency(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=1)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)
    st = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def slow_call(model, prompt, **kwargs):
        with lock:
            st["cur"] += 1
            st["max"] = max(st["max"], st["cur"])
        time.sleep(0.05)
        with lock:
            st["cur"] -= 1
        return client.CallResult(model=model, text="GUESS 의사", cost_usd=0.0,
                                 input_tokens=1, output_tokens=1, duration_ms=1,
                                 session_id="s")

    monkeypatch.setattr(client, "call", slow_call)
    parts = [f"m{i}@low" for i in range(6)]
    arena.run_arena("ko-semantle", parts, episodes=1, max_turns=1,
                    seed_base=1, run_root=tmp_path, workers=2)
    assert st["max"] <= 2      # 상한 준수
    assert st["max"] == 2      # 병렬성 실제 동작


def test_default_concurrency_capped_at_thirty_two(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=1)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)
    st = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def slow_call(model, prompt, **kwargs):
        with lock:
            st["cur"] += 1
            st["max"] = max(st["max"], st["cur"])
        time.sleep(0.05)
        with lock:
            st["cur"] -= 1
        return client.CallResult(model=model, text="GUESS 의사", cost_usd=0.0,
                                 input_tokens=1, output_tokens=1, duration_ms=1,
                                 session_id="s")

    monkeypatch.setattr(client, "call", slow_call)
    parts = [f"m{i}@low" for i in range(34)]   # 34 > 32
    arena.run_arena("ko-semantle", parts, episodes=1, max_turns=1,
                    seed_base=1, run_root=tmp_path)   # workers=None → 상한 32
    assert st["max"] <= 32
    assert st["max"] >= 19    # 19개 선택 시 전원 동시 시작 보장(순차 투입 회귀 방지)


def test_verify_detects_tampered_score(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    _patch_engine(monkeypatch, game)
    run_dir = arena.run_arena("ko-semantle", ["m-a@low"], episodes=1, max_turns=3,
                              seed_base=42, run_root=tmp_path)
    assert arena.verify_run(run_dir)["ok"] is True

    events_path = run_dir / "models" / "m-a@low" / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    tampered = []
    for line in lines:
        ev = json.loads(line)
        if ev.get("type") == "episode_end":
            ev["score"] = 0.999999
        tampered.append(json.dumps(ev, ensure_ascii=False))
    events_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")
    assert arena.verify_run(run_dir)["ok"] is False


def test_verify_handles_legacy_run(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    _patch_engine(monkeypatch, game)
    v2 = arena.run_arena("ko-semantle", ["m-a@low"], episodes=1, max_turns=3,
                         seed_base=7, run_root=tmp_path / "v2")

    # 구형 레이아웃 합성: 디렉토리=모델 id, manifest에 participants 없음
    legacy = tmp_path / "legacy" / "arena-legacy"
    (legacy / "models" / "m-a").mkdir(parents=True)
    shutil.copyfile(v2 / "models" / "m-a@low" / "events.jsonl",
                    legacy / "models" / "m-a" / "events.jsonl")
    man = json.loads((v2 / "manifest.json").read_text(encoding="utf-8"))
    man.pop("participants", None)
    man["models"] = ["m-a"]            # 구형: 모델 id 리스트
    man["effort"] = "low"
    (legacy / "manifest.json").write_text(json.dumps(man, ensure_ascii=False),
                                          encoding="utf-8")

    report = arena.verify_run(legacy)
    assert report["ok"] is True
    assert "m-a" in report["models"]    # 디렉토리(모델 id) 키로 유도


def test_verify_skips_on_game_version_mismatch(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=3)   # version 1.3.0
    _patch_engine(monkeypatch, game)
    run_dir = arena.run_arena("ko-semantle", ["m-a@low"], episodes=1, max_turns=3,
                              seed_base=5, run_root=tmp_path)
    assert arena.verify_run(run_dir)["ok"] is True     # 같은 버전 → 정상 재생

    # manifest game_version을 구버전으로 위조 → 재생하지 않고 skip
    man_path = run_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["game_version"] = "1.0.0"
    man_path.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    assert arena.verify_run(run_dir) == {
        "ok": None, "skipped": "game-version-mismatch",
        "manifest_version": "1.0.0", "current_version": "1.3.0"}


def test_verify_skips_without_ollama(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=2)
    _patch_engine(monkeypatch, game)
    run_dir = arena.run_arena("ko-semantle", ["m-a@low"], episodes=1, max_turns=2,
                              seed_base=1, run_root=tmp_path)
    monkeypatch.setattr(embed, "available", lambda: False)
    assert arena.verify_run(run_dir) == {"ok": None, "skipped": "no-ollama"}


# ----------------------------------------------------------------------
# 스트리밍: stream.json 관전 노출 + claude 스트림 파서
# ----------------------------------------------------------------------
_STREAM_KEYS = {"model", "effort", "episode", "turn", "text", "done", "updated_at"}


def test_stream_writer_sequence(tmp_path):
    path = tmp_path / "stream.json"
    w = arena._StreamWriter(path, "claude-haiku-4-5", "low", 1, 4, throttle=0.0)
    w.begin()
    s0 = json.loads(path.read_text(encoding="utf-8"))
    assert set(s0) == _STREAM_KEYS
    assert s0["text"] == "" and s0["done"] is False
    assert s0["turn"] == 4 and s0["model"] == "claude-haiku-4-5" and s0["effort"] == "low"
    w.update("부")
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "부"
    w.update("부산")
    s2 = json.loads(path.read_text(encoding="utf-8"))
    assert s2["text"] == "부산" and s2["done"] is False
    w.finish("부산 갑니다")
    s3 = json.loads(path.read_text(encoding="utf-8"))
    assert s3["text"] == "부산 갑니다" and s3["done"] is True   # 마지막 상태는 항상 기록


def test_stream_writer_resets_on_retry(tmp_path):
    path = tmp_path / "stream.json"
    w = arena._StreamWriter(path, "m", "low", 1, 2, throttle=0.0)
    w.begin()
    w.update("abc")
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "abc"
    w.begin()   # 재시도 재호출 → text=""부터 다시
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["text"] == "" and s["done"] is False


def _streaming_bot(model, prompt, *, on_text=None, **kwargs):
    import re
    m = re.search(r"현재 턴: (\d+)/", prompt)
    idx = (int(m.group(1)) - 1) if m else 0
    full = f"GUESS {_BOT_WORDS[idx % len(_BOT_WORDS)]}"
    if on_text is not None:            # 토큰 단위 누적 흉내
        acc = ""
        for ch in full:
            acc += ch
            on_text(acc)
    return client.CallResult(model=model, text=full, cost_usd=0.0, input_tokens=1,
                             output_tokens=1, duration_ms=1, session_id="s")


def test_stream_json_written_and_finalized_during_run(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=1)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(client, "call", _streaming_bot)
    run_dir = arena.run_arena("ko-semantle", ["claude-haiku-4-5@low"], episodes=1,
                              max_turns=1, seed_base=1, run_root=tmp_path)
    stream = json.loads((run_dir / "models" / "claude-haiku-4-5@low" / "stream.json")
                        .read_text(encoding="utf-8"))
    assert set(stream) == _STREAM_KEYS
    assert stream["model"] == "claude-haiku-4-5" and stream["effort"] == "low"
    assert stream["done"] is True
    assert stream["text"].startswith("GUESS")   # 수신 완료 전문
    assert "target" not in stream               # 관전 뷰에도 정답 누출 없음


class _FakePopen:
    """subprocess.Popen 대역 — 미리 준비한 stream-json 라인을 stdout로 흘린다."""

    def __init__(self, lines):
        self.stdout = iter(line + "\n" for line in lines)
        self.stderr = iter(())
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.returncode = -9


def test_claude_stream_parser_accumulates_text_delta_only(monkeypatch):
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "속으로 생각"}}}),
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "GUESS "}}}),
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "바다"}}}),
        json.dumps({"type": "result", "subtype": "success", "result": "GUESS 바다",
                    "total_cost_usd": 0.02,
                    "usage": {"input_tokens": 7, "output_tokens": 3,
                              "cache_creation_input_tokens": 4,
                              "cache_read_input_tokens": 14900},
                    "duration_ms": 55, "session_id": "sess-1", "is_error": False}),
    ]
    monkeypatch.setattr(client.subprocess, "Popen", lambda *a, **k: _FakePopen(lines))
    seen = []
    r = client.call("claude-haiku-4-5", "프롬프트", effort="low", on_text=seen.append)
    # text_delta만 누적, thinking_delta 제외
    assert seen == ["GUESS ", "GUESS 바다"]
    # 마지막 result 이벤트에서 CallResult 조립(json 모드와 동일 의미) + 캐시 사용량
    assert r.ok is True
    assert r.text == "GUESS 바다"
    assert r.cost_usd == 0.02
    assert r.input_tokens == 7 and r.output_tokens == 3
    assert r.duration_ms == 55 and r.session_id == "sess-1"
    assert r.cache_creation_input_tokens == 4
    assert r.cache_read_input_tokens == 14900


def test_claude_stream_missing_result_is_error(monkeypatch):
    lines = [json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                         "delta": {"type": "text_delta", "text": "부분"}}})]
    monkeypatch.setattr(client.subprocess, "Popen", lambda *a, **k: _FakePopen(lines))
    r = client.call("claude-haiku-4-5", "p", effort="low", on_text=lambda s: None)
    assert r.ok is False
    assert "result" in r.error or "exit" in r.error


# ----------------------------------------------------------------------
# usage 기록: CallResult 캐시 필드 + 턴 이벤트 usage 오브젝트
# ----------------------------------------------------------------------
def test_callresult_cache_fields_default_zero():
    r = client.CallResult("m", "t", 0.0, 0, 0, 0, "s")
    assert r.cache_creation_input_tokens == 0
    assert r.cache_read_input_tokens == 0


def test_claude_json_path_records_cache_usage(monkeypatch):
    payload = {"result": "GUESS 바다", "total_cost_usd": 0.01,
               "usage": {"input_tokens": 10, "output_tokens": 2,
                         "cache_creation_input_tokens": 6,
                         "cache_read_input_tokens": 26272},
               "duration_ms": 30, "session_id": "s", "is_error": False}

    class _R:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(client.subprocess, "run", lambda *a, **k: _R())
    r = client.call("claude-haiku-4-5", "p", effort="low")   # on_text=None → json 경로
    assert r.text == "GUESS 바다"
    assert r.cache_creation_input_tokens == 6
    assert r.cache_read_input_tokens == 26272


_USAGE_KEYS = {"input_tokens", "output_tokens", "cache_creation_input_tokens",
               "cache_read_input_tokens", "cost_usd", "duration_ms"}


def test_turn_event_records_usage(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=1)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)

    def fake_call(model, prompt, **kwargs):
        return client.CallResult(model=model, text="GUESS 의사", cost_usd=0.012,
                                 input_tokens=100, output_tokens=5, duration_ms=42,
                                 session_id="s", cache_creation_input_tokens=80,
                                 cache_read_input_tokens=14900)

    monkeypatch.setattr(client, "call", fake_call)
    run_dir = arena.run_arena("ko-semantle", ["m@low"], episodes=1, max_turns=1,
                              seed_base=1, run_root=tmp_path)
    events = [json.loads(l) for l in
              (run_dir / "models" / "m@low" / "events.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    turn = next(e for e in events if e["type"] == "turn")
    assert set(turn["usage"]) == _USAGE_KEYS
    assert turn["usage"] == {"input_tokens": 100, "output_tokens": 5,
                             "cache_creation_input_tokens": 80,
                             "cache_read_input_tokens": 14900,
                             "cost_usd": 0.012, "duration_ms": 42}
    # raw 전문 보존 계약 무접촉: raw는 그대로, target 누출 없음
    assert turn["raw"] == "GUESS 의사"
    assert "target" not in turn
    # 재생 검증은 usage를 무시하고 통과
    assert arena.verify_run(run_dir)["ok"] is True


# ----------------------------------------------------------------------
# 런 로버스트니스: 오라클 재시도 + 참가자 격리
# ----------------------------------------------------------------------
class _FakeEmbedResp:
    def __init__(self, vec):
        self._vec = vec

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"embeddings": [self._vec]}).encode()


def test_embed_retries_transient_failure_then_succeeds(monkeypatch):
    embed._cache.clear()
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:                       # 1·2회 실패 → 3회째 성공
            raise socket.timeout("timed out")
        return _FakeEmbedResp([0.1, 0.2])

    monkeypatch.setattr(embed.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)   # 백오프 즉시
    vecs = embed.embed(["재시도프로브"], prefix=False, model="probe")
    assert vecs == [[0.1, 0.2]]
    assert calls["n"] == 3                        # 최초 + 재시도 2회


def test_embed_raises_after_retries_exhausted(monkeypatch):
    import pytest
    embed._cache.clear()

    def always_fail(req, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(embed.urllib.request, "urlopen", always_fail)
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    with pytest.raises(OSError):
        embed.embed(["소진프로브"], prefix=False, model="probe")


class _RetryProbeOracle:
    """evaluate가 embed.embed(재시도 포함)를 경유하는 오라클."""
    words = ("학교", "의사")
    model = "probe"

    @property
    def metadata(self):
        return {"embedding_model": self.model, "reference_words": 2,
                "vocab_digest": "sha256:probe"}

    def prepare(self, target):
        return {"target": target}

    def evaluate(self, prepared, guess):
        if guess == prepared["target"]:
            return SimilarityFeedback(1.0, 1)
        embed.embed([guess], prefix=False, model=self.model)   # 네트워크 경유
        return SimilarityFeedback(0.5, 3)


def test_turn_progresses_after_oracle_retry(monkeypatch):
    embed._cache.clear()
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("timed out")     # 1회 실패
        return _FakeEmbedResp([0.1, 0.2])

    monkeypatch.setattr(embed.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    game = KoreanSemantle(_RetryProbeOracle(), max_turns=3)
    state = game.reset(1)
    ev = game.step(state, game.parse("GUESS 무지개"))   # 비타깃 → embed 경유
    assert ev["valid"] is True                          # 재시도로 흡수 → 턴 정상 진행
    assert ev["guess"] == "무지개"
    assert calls["n"] == 2                              # 1회 실패 + 1회 재시도 성공


class _PoisonOracle(FakeOracle):
    """특정 추측에서 지속적으로 죽는 오라클(재시도 소진 결과를 흉내)."""

    def evaluate(self, prepared, guess):
        if guess == "폭탄":
            raise socket.timeout("timed out")
        return super().evaluate(prepared, guess)


def test_participant_failure_is_isolated(monkeypatch, tmp_path):
    game = KoreanSemantle(_PoisonOracle(), max_turns=2)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)

    def call(model, prompt, *, on_text=None, **kwargs):
        full = f"GUESS {'폭탄' if 'bad' in model else '의사'}"
        if on_text is not None:
            on_text(full)
        return client.CallResult(model, full, 0.0, 1, 1, 1, "s")

    monkeypatch.setattr(client, "call", call)
    run_dir = arena.run_arena("ko-semantle", ["good@low", "bad@low"],
                              episodes=1, max_turns=2, seed_base=1, run_root=tmp_path)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # 부분 실패: 런 전체가 아니라 실패 참가자만 기록, status는 done
    assert manifest["status"] == "done"
    assert {f["slug"] for f in manifest.get("failed_participants", [])} == {"bad@low"}

    # 완주 참가자는 정상 마감(running으로 안 남음)
    good_live = json.loads((run_dir / "models" / "good@low" / "live.json"
                            ).read_text(encoding="utf-8"))
    assert good_live["phase"] == "done"
    assert "error" not in good_live
    good_summary = json.loads((run_dir / "models" / "good@low" / "summary.json"
                               ).read_text(encoding="utf-8"))
    assert good_summary.get("status") != "failed"

    # 실패 참가자는 live/summary에 failed + error 마킹
    bad_live = json.loads((run_dir / "models" / "bad@low" / "live.json"
                           ).read_text(encoding="utf-8"))
    assert bad_live["phase"] == "failed"
    assert bad_live["error"]
    bad_summary = json.loads((run_dir / "models" / "bad@low" / "summary.json"
                              ).read_text(encoding="utf-8"))
    assert bad_summary["status"] == "failed"
    assert bad_summary["error"]

    # verify는 완주분 검증 통과(실패분은 이벤트가 없어 무해)
    assert arena.verify_run(run_dir)["ok"] is True

    # 정답 누출 금지: error·manifest 어디에도 target 문자열 없음
    target = game.reset(1).secret
    assert target not in bad_live["error"]
    assert target not in json.dumps(manifest, ensure_ascii=False)


def test_config_pilot_wiring():
    assert config.GAME_PILOT_MODELS == ["claude-haiku-4-5", "codex-5.6-luna"]
    assert config.CODEX_MODELS["codex-5.6-luna"] == "gpt-5.6-luna"
    assert config.vendor("codex-5.6-luna") == "codex"
    assert config.alias("codex-5.6-luna") == "5.6 Luna"
