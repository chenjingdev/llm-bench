"""Mindmatch 엔진 테스트 — ollama·네트워크·실모델 호출 없이 돈다.

가짜 오라클(FakeOracle)을 KoreanSemantle에 주입하고, client.call·embed.available·
build_game을 monkeypatch로 대체해 저장 계약(v2: 참가자=모델×effort)과 재생 검증을 검사한다.
"""

import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import threading
import time

from bench import arena, client, config, embed
from bench.games import build_game, game_names
from bench.games import semantle as sm
from bench.games.base import Action, GameState
from bench.games.semantle import KoreanSemantle, SimilarityFeedback


def _guess(word: str) -> str:
    """semantle JSON 출력 프로토콜: {"발화할 단어": "<단어>"} 문자열."""
    return json.dumps({"발화할 단어": word}, ensure_ascii=False)


def _turn_no(prompt: str) -> int:
    """JSON 프롬프트에서 현재 턴 번호 추출(봇용). 기록 없으면 1."""
    m = re.search(r'"현재_턴":\s*"(\d+)/', prompt)
    return int(m.group(1)) if m else 1


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
    idx = _turn_no(prompt) - 1
    word = _BOT_WORDS[idx % len(_BOT_WORDS)]
    return client.CallResult(model=model, text=f"생각 중...\n{_guess(word)}",
                             cost_usd=0.0, input_tokens=1, output_tokens=1,
                             duration_ms=1, session_id="s")


def _patch_engine(monkeypatch, game):
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(client, "call", _bot_call)
    monkeypatch.setattr(embed, "available", lambda: True)


# ----------------------------------------------------------------------
# 게임 계층: parse / step / score / 결정론
# ----------------------------------------------------------------------
def test_parse_requires_exactly_one_json_object():
    game = KoreanSemantle(FakeOracle(), max_turns=4)
    assert game.parse("설명만 하고 JSON 없음").valid is False              # 오브젝트 0개
    assert game.parse(f"{_guess('학교')}\n{_guess('병원')}").valid is False  # 오브젝트 2개
    ok = game.parse(f"먼저 생각합니다.\n{_guess('학교')}")                 # 잡담+1개 → 유효
    assert ok.valid is True
    assert ok.value == "학교"


def test_step_rejects_duplicate_and_reports_schema():
    game = KoreanSemantle(FakeOracle(), max_turns=4)
    state = game.reset(1)
    ev = game.step(state, game.parse(_guess("의사")))
    assert ev["valid"] is True
    assert ev["guess"] == "의사"
    assert ev["rank"] == 2
    assert ev["similarity"] == round(0.5, 8)
    dup = game.step(state, game.parse(_guess("의사")))
    assert dup["valid"] is False
    assert dup["error"] == "duplicate guess"
    assert dup["guess"] == "의사"


def test_score_solved_and_unsolved():
    solved_game = KoreanSemantle(FakeOracle(), max_turns=4)
    s = solved_game.reset(11)
    solved_game.step(s, solved_game.parse(_guess(s.secret)))
    res = solved_game.result(s)
    assert res["solved"] is True
    assert res["stop_reason"] == "solved"
    assert res["score"] == 1.0

    miss_game = KoreanSemantle(FakeOracle(), max_turns=1)
    st = miss_game.reset(1)
    wrong = next(w for w in _BOT_WORDS if w != st.secret)
    miss_game.step(st, miss_game.parse(_guess(wrong)))
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


def test_registry_four_games():
    assert game_names() == ["ko-semantle", "ko-rulelab", "ko-maze", "ko-minefield"]
    assert build_game("ko-semantle").max_turns == 40
    assert build_game("ko-semantle", max_turns=12).max_turns == 12
    # 각 게임의 기본 max_turns는 자기 DEFAULT_MAX_TURNS를 따른다.
    assert build_game("ko-rulelab").max_turns == 15
    assert build_game("ko-maze").max_turns == 40


# ----------------------------------------------------------------------
# v1.1.0: 1글자 허용 · 오류 원인 구분 · 백분위 · sim_to_prev · 고착 지표
# ----------------------------------------------------------------------
def test_version_is_1_6_0():
    assert KoreanSemantle.version == "1.6.0"


def test_oracle_model_reflected_in_metadata(monkeypatch):
    # 실제 오라클: 임베딩/네트워크 없이 metadata의 embedding_model이 선택 모델을 담는지.
    monkeypatch.setattr(sm.embed, "model_info",
                        lambda m: {"name": m, "digest": "sha256:deadbeef"})
    monkeypatch.setattr(sm.embed, "embed",
                        lambda words, prefix=True, *, model=None: [[0.0]] * len(words))
    # 오라클 빌드가 워밍업(네트워크)·디스크 캐시를 타므로 둘 다 차단(네트워크·실캐시 오염 방지).
    monkeypatch.setattr(sm.embed, "warmup", lambda *a, **k: None)
    monkeypatch.setattr(sm.embed, "embed_vocab_cached",
                        lambda words, prefix=True, *, model=None: [[0.0]] * len(words))
    # 아레나 기본 오라클 모델이 honcho 태그로 이관됐다(qwen3:8b와 동일 가중치, 상주 즉답)
    assert sm.ORACLE_MODEL == "qwen3-embedding-honcho-8192"
    default_oracle = sm.EmbeddingOracle(words=("가", "나"))
    assert default_oracle.metadata["embedding_model"] == "qwen3-embedding-honcho-8192"
    assert default_oracle.metadata["embedding_digest"] == "sha256:deadbeef"
    # 명시 모델도 그대로 반영
    explicit = sm.EmbeddingOracle(words=("가", "나"), model="qwen3-embedding:4b")
    assert explicit.metadata["embedding_model"] == "qwen3-embedding:4b"
    # FakeOracle 계열은 무영향(자체 metadata 유지)
    assert FakeOracle().metadata["embedding_model"] == "fake-embedder"


def test_single_char_guess_is_valid():
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    action = game.parse(_guess("자"))         # 1글자 허용
    assert action.valid is True
    assert action.value == "자"
    state = game.reset(1)                     # '자'는 비타깃이라 해결되지 않음
    ev = game.step(state, action)
    assert ev["valid"] is True
    assert ev["guess"] == "자"


def test_parse_error_messages_distinguish_cause():
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    no_obj = game.parse("아무 JSON도 없습니다")                       # 오브젝트 0개
    two_obj = game.parse(f"{_guess('바다')}\n{_guess('하늘')}")       # 오브젝트 2개
    missing = game.parse('{"단어": "바다"}')                          # 키 누락
    non_str = game.parse('{"발화할 단어": 5}')                        # 값이 비문자열
    bad_word = game.parse('{"발화할 단어": "hello"}')                 # 값이 한국어 단어 아님
    for a in (no_obj, two_obj, missing, non_str, bad_word):
        assert a.valid is False
    assert "찾지 못" in no_obj.error                                  # 오브젝트 없음
    assert "여러 개" in two_obj.error                                 # 오브젝트 과다
    assert "키가 필요" in missing.error                               # 키 누락
    assert "문자열" in non_str.error                                  # 타입 오류
    assert "한국어 단어" in bad_word.error                            # 단어 형식 오류
    # 원인이 서로 구분된다
    assert len({no_obj.error, two_obj.error, missing.error,
                non_str.error, bad_word.error}) == 5


def test_render_is_json_with_percentile_and_output_schema():
    game = KoreanSemantle(FakeOracle(), max_turns=5)        # 어휘 4개
    state = game.reset(1)
    game.step(state, game.parse(_guess("의사")))            # rank 2 / 4 → 상위 50%
    payload = json.loads(game.render(state))                # 프롬프트는 유효 JSON
    assert payload["총_비교_어휘_수"] == 4
    rec = payload["이전_기록"][0]
    assert rec["순위"] == 2 and rec["상위백분위"] == 50 and rec["단어"] == "의사"
    assert payload["최고_순위"] == {"순위": 2, "상위백분위": 50}
    assert payload["출력_형식"] == {"발화할 단어": "<한 글자 이상의 한국어 단어 하나>"}
    assert isinstance(payload["time"], int)                 # time은 정수 필드
    # time에는 어떤 설명·지시도 붙지 않는다("무시"/"난수"/"세션" 문구 금지 — 실험 변인)
    blob = game.render(state)
    for banned in ("무시", "난수", "세션", "식별"):
        assert banned not in blob


def test_render_prefix_stable_for_cache_alignment():
    # JSON 직렬화 순서: [정적 규칙+time] → [이전_기록(append-only)] → [변동부].
    # 연속 두 턴의 공통 prefix가 정적부+time+기존 기록 원소를 포함하고, 변동부
    # (현재_턴/최고_순위)는 그 뒤에만 와야 캐시가 정렬된다. time은 에피소드 내 불변.
    game = KoreanSemantle(FakeOracle(), max_turns=10)
    state = game.reset(1)
    game.step(state, game.parse(_guess("활동")))    # 비타깃(rank 3) → 해결 안 됨
    p_k = game.render(state)                         # 기록 1행
    game.step(state, game.parse(_guess("생각")))
    p_k1 = game.render(state)                        # 기록 2행(append-only 연장)

    common = os.path.commonprefix([p_k, p_k1])
    assert "매 응답에는 정확히 한 개의 추측만 담으세요." in common     # 고정 규칙
    assert '"time":' in common                        # time은 정적 prefix에 포함(불변)
    assert '"이전_기록":' in common
    assert '"단어": "활동"' in common                 # 기존 기록 원소가 prefix에 그대로
    # 변동부는 공통 prefix에 없다(divergence 뒤에만)
    assert '"현재_턴":' not in common
    assert '"최고_순위":' not in common
    # 그리고 각 프롬프트에서 변동부는 공통 prefix 뒤 꼬리에 존재
    assert '"현재_턴":' in p_k[len(common):]
    assert '"현재_턴":' in p_k1[len(common):]
    # 직렬화 순서 확인: time·이전_기록이 현재_턴보다 앞
    assert p_k.index('"time":') < p_k.index('"이전_기록":') < p_k.index('"현재_턴":')


def test_sim_to_prev_is_deterministic():
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    guesses = ["생활", "생각", "활동"]                       # 전부 비타깃 → 해결 안 됨

    def run():
        s = game.reset(1)
        return [game.step(s, game.parse(_guess(w))) for w in guesses]

    a, b = run(), run()
    assert a == b                                            # 결정론(재생 대조 대상)
    assert a[0]["sim_to_prev"] is None                       # 첫 유효 추측
    assert a[1]["sim_to_prev"] == round(1 / 3, 8)            # 생활→생각: 공유 '생'
    assert a[2]["sim_to_prev"] == 0.0                        # 생각→활동: 공유 없음


def test_fixation_metrics_max_plateau_and_sim():
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    state = game.reset(3)
    for w in ["생활", "생각", "활동"]:                       # 모두 rank 3(비타깃)
        game.step(state, game.parse(_guess(w)))
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
    game.step(state, game.parse(_guess("활동")))               # rank 3 (비타깃)
    game.step(state, game.parse(_guess("의사")))               # rank 2 (개선)
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
                                "median_fixation_sim", "invalid_actions", "usage"}
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
        return client.CallResult(model=model, text=_guess("의사"), cost_usd=0.0,
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
        return client.CallResult(model=model, text=_guess("의사"), cost_usd=0.0,
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
        return client.CallResult(model=model, text=_guess("의사"), cost_usd=0.0,
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
            # score는 오라클 노이즈 허용 필드(5e-3)이므로 위변조 탐지엔 임계를 크게 넘는
            # 조작이어야 한다 — 실제 저장값에서 0.5만큼 밀어 노이즈와 명확히 구분한다.
            ev["score"] = round(ev["score"] + 0.5, 6)
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
    game = KoreanSemantle(FakeOracle(), max_turns=3)   # version 1.6.0
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
        "manifest_version": "1.0.0", "current_version": "1.6.0"}


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
    idx = _turn_no(prompt) - 1
    full = _guess(_BOT_WORDS[idx % len(_BOT_WORDS)])
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
    assert "발화할 단어" in stream["text"]        # 수신 완료 전문(JSON 출력)
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

    # json 경로는 이제 프로세스-그룹 kill 안전한 client._run을 거친다(subprocess.run 대체).
    monkeypatch.setattr(client, "_run", lambda *a, **k: _R())
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
        return client.CallResult(model=model, text=_guess("의사"), cost_usd=0.012,
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
    assert turn["raw"] == _guess("의사")
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
    calls["n"] = 0   # reset의 시작_기록 채점분 제외 — 이제부터 추측 턴만 계량(실패 1회 재장전)
    ev = game.step(state, game.parse(_guess("무지개")))   # 비타깃 → embed 경유
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
        full = _guess("폭탄" if "bad" in model else "의사")
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


# ----------------------------------------------------------------------
# 멀티게임 일반화(계약 v1 §1·§2): 임의 TURN_FIELDS/RESULT_FIELDS를 가진 스텁
# 게임이 엔진 코드 수정 없이 저장 스키마·재생 검증에 태워지는지 검사한다.
# ----------------------------------------------------------------------
class StubGame:
    """오라클리스 스텁 게임 — 'PICK <n>' 한 줄로 secret(=seed%10)에 근접.

    필드명을 semantle과 전혀 다르게(pick/delta/closest/tries) 잡아, 엔진이
    game.TURN_FIELDS/INVALID_KEEP/LIVE_LAST_FIELDS/RESULT_FIELDS/progress/
    summary_stats만으로 events/live/summary/verify를 조립하는지 증명한다.
    needs_ollama=False → ollama 없이도 재생 검증이 수행되어야 한다.
    """
    id = "ko-stub"
    version = "9.9.9"
    TURN_FIELDS = ("pick", "delta")          # semantle과 다른 필드 집합
    INVALID_KEEP = ("pick",)                 # 중복 무효 턴에서 pick 보존
    LIVE_LAST_FIELDS = ("pick", "delta")
    RESULT_FIELDS = ("solved", "turns", "score", "closest")
    needs_ollama = False

    def __init__(self, max_turns=3):
        self.max_turns = max_turns

    @property
    def metadata(self):
        return {"game": self.id, "version": self.version, "kind": "stub"}

    def reset(self, seed):
        return GameState(self.id, self.version, seed, self.max_turns, str(seed % 10))

    def render(self, state):
        return (f"규칙 고정 블록\n현재 턴: {state.turn + 1}/{state.max_turns}\n"
                "PICK <n>")

    def parse(self, text):
        import re
        m = re.search(r"PICK\s+(-?\d+)", text or "")
        if not m:
            return Action("pick", "", text or "", False, "PICK 한 줄 필요")
        return Action("pick", m.group(1), text or "")

    def step(self, state, action):
        state.turn += 1
        if not action.valid:
            ev = {"turn": state.turn, "valid": False, "raw": action.raw,
                  "error": action.error}
        elif action.value in state.seen:
            ev = {"turn": state.turn, "valid": False, "raw": action.raw,
                  "pick": action.value, "error": "duplicate"}
        else:
            state.seen.add(action.value)
            delta = abs(int(action.value) - int(state.secret))
            ev = {"turn": state.turn, "valid": True, "raw": action.raw,
                  "pick": action.value, "delta": delta}
            if delta == 0:
                state.solved = True
                state.done = True
                state.stop_reason = "solved"
        state.history.append(ev)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return ev

    def result(self, state):
        deltas = [e["delta"] for e in state.history if e.get("valid") and "delta" in e]
        return {"solved": state.solved, "turns": state.turn,
                "score": 1.0 if state.solved else 0.0,
                "closest": min(deltas) if deltas else None,
                "stop_reason": state.stop_reason}

    def progress(self, state):
        deltas = [e["delta"] for e in state.history if e.get("valid") and "delta" in e]
        return {"closest": min(deltas) if deltas else None, "tries": len(deltas)}

    def summary_stats(self, episode_ends):
        closes = [e["closest"] for e in episode_ends if e.get("closest") is not None]
        return {"median_closest": statistics.median(closes) if closes else None}


def _stub_bot(model, prompt, *, on_text=None, **kwargs):
    """턴1 유효 PICK, 턴2 동일 PICK(중복), 턴3 행동 없음(파싱 실패)."""
    import re
    m = re.search(r"현재 턴: (\d+)/", prompt)
    t = int(m.group(1)) if m else 1
    text = "생각만 하고 행동은 없음" if t == 3 else "PICK 3"
    if on_text is not None:
        on_text(text)
    return client.CallResult(model=model, text=text, cost_usd=0.0, input_tokens=1,
                             output_tokens=1, duration_ms=1, session_id="s")


def test_generalized_engine_assembles_arbitrary_game(monkeypatch, tmp_path):
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)   # 비오라클 게임 → ollama 불필요
    monkeypatch.setattr(client, "call", _stub_bot)

    run_dir = arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=3,
                              effort="low", seed_base=100, run_root=tmp_path)

    slug = run_dir / "models" / "s-a@low"
    events = [json.loads(l) for l in
              (slug / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    turns = [e for e in events if e["type"] == "turn"]
    ends = [e for e in events if e["type"] == "episode_end"]

    # 턴 레코드: 기본 키 + valid→TURN_FIELDS + progress(closest,tries) + raw/ts/usage.
    valid = next(e for e in turns if e["valid"])
    assert set(valid) == {"type", "episode", "turn", "valid", "pick", "delta",
                          "closest", "tries", "raw", "ts", "usage"}
    assert "target" not in valid                       # 정답 누출 금지
    assert valid["pick"] == "3"

    # 무효(파싱 실패): TURN_FIELDS 미포함, error + progress + raw/ts.
    parsefail = next(e for e in turns if not e["valid"] and "pick" not in e)
    assert set(parsefail) == {"type", "episode", "turn", "valid", "error",
                              "closest", "tries", "raw", "ts", "usage"}

    # 무효(중복): INVALID_KEEP=("pick",) → pick 보존, delta는 없음.
    dup = next(e for e in turns if not e["valid"] and "pick" in e)
    assert dup["pick"] == "3" and "delta" not in dup
    assert dup["error"] == "duplicate"

    # episode_end: {type, episode, target, *RESULT_FIELDS, stop_reason, nonce, ts}.
    # target은 여기에만. nonce(에피소드 인스턴스 태그)는 추가 필드.
    assert len(ends) == 2
    for e in ends:
        assert set(e) == {"type", "episode", "target", "solved", "turns",
                          "score", "closest", "stop_reason", "nonce", "ts"}
    assert ends[0]["target"] == "0"                    # seed 100 → secret "0"
    assert ends[1]["target"] == "1"                    # seed 101 → secret "1"

    # live.json: 공통부 + last_pick/last_delta + progress(closest,tries) + 스니펫.
    live = json.loads((slug / "live.json").read_text(encoding="utf-8"))
    assert set(live) == {"model", "effort", "episode", "turn", "max_turns", "phase",
                         "last_pick", "last_delta", "closest", "tries",
                         "raw_snippet", "updated_at"}
    assert "target" not in live
    assert live["last_pick"] == ""                     # 마지막 턴 파싱 실패 → 첫 필드 "" 기본
    assert live["last_delta"] is None                  # 부속 필드는 None 기본

    # summary.json: 공통부 + summary_stats(median_closest).
    summary = json.loads((slug / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"model", "effort", "episodes", "mean_score", "solve_rate",
                            "median_turns", "median_closest", "invalid_actions", "usage"}
    assert summary["invalid_actions"] == 4             # 에피소드당 무효 2개(중복+파싱) × 2

    # needs_ollama=False → embed.available()=False여도 재생 검증이 돌아 통과한다.
    assert arena.verify_run(run_dir)["ok"] is True


def _semantle_mix_bot(model, prompt, *, on_text=None, **kwargs):
    """턴1 유효(비타깃 '무지개'), 턴2 중복, 턴3 파싱 실패 — 세 갈래 턴 레코드 생성."""
    t = _turn_no(prompt)
    text = "행동 없이 설명만" if t == 3 else _guess("무지개")
    if on_text is not None:
        on_text(text)
    return client.CallResult(model=model, text=text, cost_usd=0.0, input_tokens=1,
                             output_tokens=1, duration_ms=1, session_id="s")


def test_semantle_schema_regression(monkeypatch, tmp_path):
    # 일반화 후에도 ko-semantle 산출 파일의 키 집합·핵심 값이 기존과 동일해야 한다.
    game = KoreanSemantle(FakeOracle(), max_turns=3)   # '무지개'는 비타깃 → 미해결
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(client, "call", _semantle_mix_bot)

    run_dir = arena.run_arena("ko-semantle", ["m@low"], episodes=1, max_turns=3,
                              effort="low", seed_base=100, run_root=tmp_path)
    slug = run_dir / "models" / "m@low"
    events = [json.loads(l) for l in
              (slug / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    turns = [e for e in events if e["type"] == "turn"]
    ends = [e for e in events if e["type"] == "episode_end"]

    valid = next(e for e in turns if e["valid"])
    assert set(valid) == {"type", "episode", "turn", "valid", "guess", "similarity",
                          "rank", "sim_to_prev", "best_rank", "raw", "ts", "usage"}
    assert valid["best_rank"] == 3                     # progress로 산출해도 유효 rank와 일치

    dup = next(e for e in turns if not e["valid"] and "guess" in e)
    assert set(dup) == {"type", "episode", "turn", "valid", "guess", "error",
                        "best_rank", "raw", "ts", "usage"}
    assert dup["error"] == "duplicate guess"

    parsefail = next(e for e in turns if not e["valid"] and "guess" not in e)
    assert set(parsefail) == {"type", "episode", "turn", "valid", "error",
                              "best_rank", "raw", "ts", "usage"}

    assert set(ends[0]) == {"type", "episode", "target", "solved", "turns", "best_rank",
                            "best_rank_curve", "score", "max_plateau", "fixation_sim",
                            "시작_기록", "stop_reason", "nonce", "ts"}
    assert "target" in ends[0]

    live = json.loads((slug / "live.json").read_text(encoding="utf-8"))
    assert set(live) == _LIVE_KEYS
    assert live["last_guess"] == ""                    # 마지막 턴 파싱 실패 → "" 보존(값 동일성)
    assert live["last_similarity"] is None

    summary = json.loads((slug / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"model", "effort", "episodes", "mean_score", "solve_rate",
                            "median_turns", "median_max_plateau", "median_fixation_sim",
                            "invalid_actions", "usage"}

    assert arena.verify_run(run_dir)["ok"] is True


# ----------------------------------------------------------------------
# verify 검증 정책: 무효 턴 progress 파생 필드 제외 + TOLERANT_FIELDS 오차 허용
# ----------------------------------------------------------------------
class OverlapStub:
    """progress()가 TURN_FIELDS와 겹치는 필드(dist)를 병합하는 스텁(maze 재현).

    무효 턴에는 step 이벤트에 dist가 없지만 progress가 dist를 병합해 저장 레코드엔
    dist가 남는다. verify가 무효 턴에서 TURN_FIELDS(dist)를 재생 step 이벤트와
    대조하면 오탐이 난다 → valid 턴에만 대조해야 통과한다.
    """
    id = "ko-overlap"
    version = "1.0.0"
    TURN_FIELDS = ("move", "dist")
    INVALID_KEEP = ()
    LIVE_LAST_FIELDS = ("move", "dist")
    RESULT_FIELDS = ("solved", "turns", "score", "dist")
    needs_ollama = False
    TOLERANT_FIELDS = {}

    def __init__(self, max_turns=4):
        self.max_turns = max_turns

    @property
    def metadata(self):
        return {"game": self.id, "version": self.version}

    def reset(self, seed):
        return GameState(self.id, self.version, seed, self.max_turns, str(seed))

    def render(self, state):
        return f"고정 블록\n현재 턴: {state.turn + 1}/{state.max_turns}\nMOVE <dir>"

    def parse(self, text):
        import re
        m = re.search(r"MOVE\s+(\S+)", text or "")
        if not m:
            return Action("move", "", text or "", False, "MOVE 필요")
        return Action("move", m.group(1), text or "")

    def _cur_dist(self, state):
        return max(0, 3 - sum(1 for e in state.history if e.get("valid")))

    def step(self, state, action):
        state.turn += 1
        if not action.valid:
            ev = {"turn": state.turn, "valid": False, "raw": action.raw,
                  "error": action.error}
        else:
            moves = sum(1 for e in state.history if e.get("valid")) + 1
            dist = max(0, 3 - moves)
            ev = {"turn": state.turn, "valid": True, "raw": action.raw,
                  "move": action.value, "dist": dist}
            if dist == 0:
                state.solved = True
                state.done = True
                state.stop_reason = "solved"
        state.history.append(ev)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return ev

    def progress(self, state):
        # dist는 TURN_FIELDS와 겹친다 — valid 턴엔 step 값과 동일, 무효 턴엔 현재값 유지.
        return {"dist": self._cur_dist(state),
                "steps": sum(1 for e in state.history if e.get("valid"))}

    def result(self, state):
        return {"solved": state.solved, "turns": state.turn,
                "score": 1.0 if state.solved else 0.0, "dist": self._cur_dist(state),
                "stop_reason": state.stop_reason}

    def summary_stats(self, episode_ends):
        return {}


def _overlap_bot(model, prompt, *, on_text=None, **kwargs):
    """턴2에 형식 오류(무효)를 끼워 progress 파생 dist가 무효 레코드에 남게 한다."""
    import re
    m = re.search(r"현재 턴: (\d+)/", prompt)
    t = int(m.group(1)) if m else 1
    text = "행동 없음" if t == 2 else "MOVE 북"
    if on_text is not None:
        on_text(text)
    return client.CallResult(model=model, text=text, cost_usd=0.0, input_tokens=1,
                             output_tokens=1, duration_ms=1, session_id="s")


def test_verify_ignores_progress_fields_on_invalid_turns(monkeypatch, tmp_path):
    game = OverlapStub(max_turns=4)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    monkeypatch.setattr(client, "call", _overlap_bot)

    run_dir = arena.run_arena("ko-overlap", ["o@low"], episodes=1, max_turns=4,
                              effort="low", seed_base=1, run_root=tmp_path)
    slug = run_dir / "models" / "o@low"
    events = [json.loads(l) for l in
              (slug / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    # 무효 턴 레코드에 progress 파생 dist가 실제로 남아 있어야 시나리오가 성립한다.
    invalid = next(e for e in events if e["type"] == "turn" and not e["valid"])
    assert "dist" in invalid and "move" not in invalid

    # 그럼에도 verify는 무효 턴에서 TURN_FIELDS(dist)를 대조하지 않으므로 통과한다.
    assert arena.verify_run(run_dir)["ok"] is True


class TolStub:
    """TOLERANT_FIELDS(스칼라 val, 리스트 curve, score)를 가진 스텁."""
    id = "ko-tol"
    version = "1.0.0"
    TURN_FIELDS = ("val",)
    INVALID_KEEP = ()
    LIVE_LAST_FIELDS = ("val",)
    RESULT_FIELDS = ("solved", "turns", "score", "curve")
    needs_ollama = False
    TOLERANT_FIELDS = {"val": 0.01, "curve": 0.01, "score": 0.01}

    def __init__(self, max_turns=3):
        self.max_turns = max_turns

    @property
    def metadata(self):
        return {"game": self.id, "version": self.version}

    def reset(self, seed):
        return GameState(self.id, self.version, seed, self.max_turns, str(seed))

    def render(self, state):
        return f"블록\n현재 턴: {state.turn + 1}/{state.max_turns}\nGO"

    def parse(self, text):
        if "GO" in (text or ""):
            return Action("go", "go", text or "")
        return Action("go", "", text or "", False, "GO 필요")

    def step(self, state, action):
        state.turn += 1
        if not action.valid:
            ev = {"turn": state.turn, "valid": False, "raw": action.raw,
                  "error": action.error}
        else:
            ev = {"turn": state.turn, "valid": True, "raw": action.raw,
                  "val": round(1.0 / state.turn, 6)}
        state.history.append(ev)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return ev

    def result(self, state):
        curve = [e["val"] for e in state.history if e.get("valid")]
        return {"solved": False, "turns": state.turn,
                "score": round(sum(curve), 6), "curve": curve,
                "stop_reason": state.stop_reason}

    def progress(self, state):
        return {}

    def summary_stats(self, episode_ends):
        return {}


def test_verify_tolerant_fields_absorb_noise_but_catch_tampering(monkeypatch, tmp_path):
    game = TolStub(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    monkeypatch.setattr(client, "call", lambda model, prompt, **kw:
                        client.CallResult(model, "GO", 0.0, 1, 1, 1, "s"))

    run_dir = arena.run_arena("ko-tol", ["t@low"], episodes=1, max_turns=3,
                              effort="low", seed_base=1, run_root=tmp_path)
    events_path = run_dir / "models" / "t@low" / "events.jsonl"
    pristine = events_path.read_text(encoding="utf-8")
    assert arena.verify_run(run_dir)["ok"] is True          # 청정 런은 통과

    def tamper(mutator) -> bool:
        out = []
        for line in pristine.splitlines():
            ev = json.loads(line)
            mutator(ev)
            out.append(json.dumps(ev, ensure_ascii=False))
        events_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return arena.verify_run(run_dir)["ok"]

    def scalar(delta):
        def m(ev):
            if ev.get("type") == "turn" and ev.get("valid") and "val" in ev:
                ev["val"] = round(ev["val"] + delta, 8)
        return m

    def curve_shift(delta):
        def m(ev):
            if ev.get("type") == "episode_end":
                ev["curve"] = [round(x + delta, 8) for x in ev["curve"]]
        return m

    def curve_grow(ev):
        if ev.get("type") == "episode_end":
            ev["curve"] = ev["curve"] + [0.0]

    # 스칼라 tolerant(val, tol 0.01): 오차 이내는 통과, 초과는 탐지.
    assert tamper(scalar(0.005)) is True
    assert tamper(scalar(0.02)) is False
    # 리스트 tolerant(curve, tol 0.01): 원소별 오차 이내는 통과, 초과·길이 변경은 탐지.
    assert tamper(curve_shift(0.005)) is True
    assert tamper(curve_shift(0.02)) is False
    assert tamper(curve_grow) is False


# ----------------------------------------------------------------------
# 결과 재사용(측정 경제, 계약 §9): measurement_key + 동일 조건 완주분 편입
# ----------------------------------------------------------------------
class StubGameV2(StubGame):
    """게임 버전만 다른 스텁 — measurement_key(oracle·game_version) 변화를 검증."""
    version = "9.9.10"


class PoisonStub(StubGame):
    """특정 pick("7")에서 step이 죽는 스텁 — 참가자 실패 격리·재사용 제외 검증."""
    def step(self, state, action):
        if action.valid and action.value == "7":
            raise RuntimeError("poison")
        return super().step(state, action)


def _counting(calls, inner):
    def call(model, prompt, **kw):
        calls["n"] += 1
        return inner(model, prompt, **kw)
    return call


def _seq_stamps(monkeypatch):
    """run_id 충돌 방지 — _stamp는 초 해상도라 같은 초의 두 런이 겹칠 수 있다."""
    seq = {"n": 0}

    def fake():
        seq["n"] += 1
        return f"arena-seq-{seq['n']}"

    monkeypatch.setattr(arena, "_stamp", fake)


def test_measurement_key_reflects_conditions():
    base = {"game": "g", "game_version": "1", "oracle": {"a": 1},
            "seeds": [1, 2], "max_turns": 3}
    assert arena.measurement_key(base) == arena.measurement_key(dict(base))   # 결정론
    for field, val in [("seeds", [1, 2, 3]), ("oracle", {"a": 2}),
                       ("max_turns", 4), ("game_version", "2"), ("game", "h")]:
        other = dict(base)
        other[field] = val
        assert arena.measurement_key(base) != arena.measurement_key(other)
    assert config.ARENA_SUITE_SEED_BASE == 314159


def test_reuse_skips_measurement_for_matching_run(monkeypatch, tmp_path):
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    _seq_stamps(monkeypatch)
    calls = {"n": 0}
    monkeypatch.setattr(client, "call", _counting(calls, _stub_bot))

    run1 = arena.run_arena("ko-stub", ["s-a@low", "s-b@low"], episodes=2, max_turns=3,
                           effort="low", seed_base=100, run_root=tmp_path)
    assert calls["n"] > 0
    man1 = json.loads((run1 / "manifest.json").read_text(encoding="utf-8"))

    calls["n"] = 0
    run2 = arena.run_arena("ko-stub", ["s-a@low", "s-b@low"], episodes=2, max_turns=3,
                           effort="low", seed_base=100, run_root=tmp_path)
    assert calls["n"] == 0                       # 재측정 없음(모델 호출 0)

    man2 = json.loads((run2 / "manifest.json").read_text(encoding="utf-8"))
    assert "measurement_key" in man2
    assert man1["measurement_key"] == man2["measurement_key"]
    for p in man2["participants"]:
        assert p["reused_from"] == run1.name     # 원본 run_id 기록
    # 복사된 events/summary는 원본과 바이트 동일, stream.json은 복사 제외.
    for slug in ("s-a@low", "s-b@low"):
        for fname in ("events.jsonl", "summary.json"):
            src = (run1 / "models" / slug / fname).read_text(encoding="utf-8")
            dst = (run2 / "models" / slug / fname).read_text(encoding="utf-8")
            assert src == dst
        assert not (run2 / "models" / slug / "stream.json").exists()
    # 재사용 런도 재생 검증을 그대로 통과한다(seeds 동일).
    assert arena.verify_run(run2)["ok"] is True


def test_reuse_skipped_on_key_mismatch(monkeypatch, tmp_path):
    _seq_stamps(monkeypatch)
    monkeypatch.setattr(embed, "available", lambda: False)
    calls = {"n": 0}
    monkeypatch.setattr(client, "call", _counting(calls, _stub_bot))

    g3 = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: g3)
    arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=3, effort="low",
                    seed_base=100, run_root=tmp_path)

    # seed_base 다름 → seeds 다름 → 키 다름 → 재측정
    calls["n"] = 0
    arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=3, effort="low",
                    seed_base=777, run_root=tmp_path)
    assert calls["n"] > 0

    # max_turns 다름 → 키 다름 → 재측정
    calls["n"] = 0
    g5 = StubGame(max_turns=5)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: g5)
    arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=5, effort="low",
                    seed_base=100, run_root=tmp_path)
    assert calls["n"] > 0

    # game_version(=oracle 포함) 다름 → 키 다름 → 재측정
    calls["n"] = 0
    gv = StubGameV2(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: gv)
    r = arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=3, effort="low",
                        seed_base=100, run_root=tmp_path)
    assert calls["n"] > 0
    man = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
    assert "reused_from" not in man["participants"][0]


def test_reuse_excludes_failed_participants(monkeypatch, tmp_path):
    game = PoisonStub(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    _seq_stamps(monkeypatch)

    def bot(model, prompt, *, on_text=None, **kw):
        text = "PICK 7" if "bad" in model else "PICK 3"   # bad는 poison pick → 실패
        if on_text is not None:
            on_text(text)
        return client.CallResult(model, text, 0.0, 1, 1, 1, "s")

    monkeypatch.setattr(client, "call", bot)

    run1 = arena.run_arena("ko-stub", ["good@low", "bad@low"], episodes=2, max_turns=3,
                           effort="low", seed_base=100, run_root=tmp_path)
    man1 = json.loads((run1 / "manifest.json").read_text(encoding="utf-8"))
    assert {f["slug"] for f in man1.get("failed_participants", [])} == {"bad@low"}

    run2 = arena.run_arena("ko-stub", ["good@low", "bad@low"], episodes=2, max_turns=3,
                           effort="low", seed_base=100, run_root=tmp_path)
    man2 = json.loads((run2 / "manifest.json").read_text(encoding="utf-8"))
    parts = {p["slug"]: p for p in man2["participants"]}
    assert parts["good@low"].get("reused_from") == run1.name   # 완주분만 재사용
    assert "reused_from" not in parts["bad@low"]                # 실패분은 재측정
    assert {f["slug"] for f in man2.get("failed_participants", [])} == {"bad@low"}


def test_reuse_disabled_forces_remeasurement(monkeypatch, tmp_path):
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    _seq_stamps(monkeypatch)
    calls = {"n": 0}
    monkeypatch.setattr(client, "call", _counting(calls, _stub_bot))

    arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=3, effort="low",
                    seed_base=100, run_root=tmp_path)
    calls["n"] = 0
    run2 = arena.run_arena("ko-stub", ["s-a@low"], episodes=2, max_turns=3, effort="low",
                           seed_base=100, run_root=tmp_path, reuse=False)
    assert calls["n"] > 0                        # 강제 재측정
    man2 = json.loads((run2 / "manifest.json").read_text(encoding="utf-8"))
    assert "reused_from" not in man2["participants"][0]
    assert "measurement_key" in man2             # 키는 재사용 여부와 무관하게 기록


def test_reuse_excludes_model_error_episodes(monkeypatch, tmp_path):
    # quota/호출 실패로 끝난 에피소드는 "완주"처럼 보여도(summary status 없음, episode_end
    # 수 일치) model_error 종료라 영구 결과로 재사용하면 안 된다 → 재측정되어야 한다.
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    _seq_stamps(monkeypatch)
    calls = {"n": 0}

    def failing(model, prompt, *, on_text=None, **kw):
        calls["n"] += 1
        return client.CallResult(model=model, text="", cost_usd=0.0, input_tokens=0,
                                 output_tokens=0, duration_ms=0, session_id="s",
                                 ok=False, error="quota")

    monkeypatch.setattr(client, "call", failing)

    run1 = arena.run_arena("ko-stub", ["m@low"], episodes=2, max_turns=3, effort="low",
                           seed_base=100, run_root=tmp_path, retries=0)
    # 구식 완주 기준은 충족(summary에 status 없음, episode_end 2개)…
    s1 = json.loads((run1 / "models" / "m@low" / "summary.json").read_text(encoding="utf-8"))
    assert "status" not in s1
    ends = [json.loads(l) for l in
            (run1 / "models" / "m@low" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(l)["type"] == "episode_end"]
    assert len(ends) == 2
    assert all(e["stop_reason"] == "model_error" for e in ends)   # …그러나 전부 model_error

    calls["n"] = 0
    run2 = arena.run_arena("ko-stub", ["m@low"], episodes=2, max_turns=3, effort="low",
                           seed_base=100, run_root=tmp_path, retries=0)
    assert calls["n"] > 0                        # 재사용 안 하고 재측정
    man2 = json.loads((run2 / "manifest.json").read_text(encoding="utf-8"))
    assert "reused_from" not in man2["participants"][0]


def test_summary_usage_sums_turn_usage_including_zero_model_error(monkeypatch, tmp_path):
    # 참가자 summary.usage = 전 턴 이벤트 usage 합계. 성공 턴은 고정 usage, model_error
    # 턴(호출 실패)은 usage 0이라 합계에 0으로 포함된다.
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    seen = {"n": 0}

    def bot(model, prompt, *, on_text=None, **kw):
        seen["n"] += 1
        if seen["n"] >= 4:            # 에피소드2 첫 호출부터 실패 → model_error 턴(usage 0)
            return client.CallResult(model=model, text="", cost_usd=0.0, input_tokens=0,
                                     output_tokens=0, duration_ms=0, session_id="s",
                                     ok=False, error="quota")
        if on_text is not None:
            on_text("PICK 3")
        return client.CallResult(model=model, text="PICK 3", cost_usd=0.001,
                                 input_tokens=10, output_tokens=2, duration_ms=5,
                                 session_id="s", cache_creation_input_tokens=3,
                                 cache_read_input_tokens=100)

    monkeypatch.setattr(client, "call", bot)
    run_dir = arena.run_arena("ko-stub", ["m@low"], episodes=2, max_turns=3,
                              effort="low", seed_base=100, run_root=tmp_path, retries=0)
    slug = run_dir / "models" / "m@low"

    events = [json.loads(l) for l in
              (slug / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    turns = [e for e in events if e["type"] == "turn"]
    ok_turns = [e for e in turns if e["usage"]["input_tokens"] == 10]
    err_turns = [e for e in turns if e["usage"]["input_tokens"] == 0]
    assert len(ok_turns) == 3           # 에피소드1의 성공 3턴
    assert len(err_turns) == 1          # 에피소드2의 model_error 턴
    assert err_turns[0]["usage"] == arena._zero_usage()   # model_error 턴 usage=0

    summary = json.loads((slug / "summary.json").read_text(encoding="utf-8"))
    assert "status" not in summary                        # 참가자는 완료(성공 경로)
    assert set(summary["usage"]) == _USAGE_KEYS
    # 성공 3턴 × 고정 usage + model_error 0 = 합계(cost_usd는 round 6).
    assert summary["usage"] == {"input_tokens": 30, "output_tokens": 6,
                                "cache_creation_input_tokens": 9,
                                "cache_read_input_tokens": 300,
                                "cost_usd": round(0.003, 6), "duration_ms": 15}


# ----------------------------------------------------------------------
# 준비 단계 가시화(계약 부록 B): build 전 예비 manifest + 빌드 실패 정직성
# ----------------------------------------------------------------------
_FINAL_MANIFEST_KEYS = {"run_id", "game", "participants", "models", "episodes",
                        "max_turns", "effort", "status", "started_at", "finished_at",
                        "pid", "game_version", "oracle", "seeds", "verify",
                        "measurement_key"}


def test_preliminary_manifest_visible_during_build(monkeypatch, tmp_path):
    game = StubGame(max_turns=3)
    seen = {}

    def slow_build(name, *, max_turns=None):
        # verify_run도 build_game을 호출하므로 최초(예비 빌드) 호출만 관측한다.
        if "manifest_status" not in seen:
            # 빌드 시점(콜드 스타트 구간)에 이미 예비 manifest가 index·파일에 있어야 한다.
            idx = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
            entry = idx["runs"][0]
            seen["index_status"] = entry["status"]
            man = json.loads((tmp_path / entry["run_id"] / "manifest.json")
                             .read_text(encoding="utf-8"))
            seen["manifest_status"] = man["status"]
            seen["max_turns"] = man["max_turns"]        # 요청값 그대로(None이면 null)
            seen["omitted"] = [k for k in ("game_version", "oracle", "measurement_key",
                                           "verify") if k not in man]
        return game

    monkeypatch.setattr(arena, "build_game", slow_build)
    monkeypatch.setattr(embed, "available", lambda: False)
    monkeypatch.setattr(client, "call", _stub_bot)

    run_dir = arena.run_arena("ko-stub", ["s@low"], episodes=2, max_turns=None,
                              effort="low", seed_base=100, run_root=tmp_path)

    # 빌드 시점: preparing + 요청 max_turns(None→null) + 미지값 필드 4종 생략
    assert seen["manifest_status"] == "preparing"
    assert seen["index_status"] == "preparing"
    assert seen["max_turns"] is None
    assert set(seen["omitted"]) == {"game_version", "oracle", "measurement_key", "verify"}

    # 완주 후: 최종 manifest는 오늘과 동일한 필드셋 + status done + 해소된 값.
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(man) == _FINAL_MANIFEST_KEYS
    assert man["status"] == "done"
    assert man["max_turns"] == 3                        # game.max_turns로 해소
    assert man["game_version"] == game.version
    assert man["oracle"] == game.metadata
    assert man["measurement_key"]
    assert man["verify"]["ok"] is True
    idx = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert idx["runs"][0]["status"] == "done"


def test_build_failure_marks_manifest_failed_and_raises(monkeypatch, tmp_path):
    import pytest

    def boom(name, *, max_turns=None):
        raise OSError("ollama 콜드 스타트 실패")

    monkeypatch.setattr(arena, "build_game", boom)
    monkeypatch.setattr(embed, "available", lambda: False)

    with pytest.raises(OSError):
        arena.run_arena("ko-stub", ["s@low"], episodes=2, max_turns=3,
                        effort="low", seed_base=100, run_root=tmp_path)

    # preparing 잔재가 아니라 failed로 마감(+error+finished_at) — 예외도 전파된다.
    idx = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    entry = idx["runs"][0]
    assert entry["status"] == "failed"                 # index에도 failed 반영
    man = json.loads((tmp_path / entry["run_id"] / "manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["status"] == "failed"
    assert "ollama 콜드 스타트 실패" in man["error"]
    assert man["finished_at"] is not None
    assert "measurement_key" not in man                # 빌드 전이라 아직 없음


# ----------------------------------------------------------------------
# 같은 시드 반복 측정(repeat_seed): 같은 문제를 N회 풀어 표본 편차를 잰다
# ----------------------------------------------------------------------
def test_repeat_seed_same_problem_each_episode(monkeypatch, tmp_path):
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    monkeypatch.setattr(client, "call", _stub_bot)

    run_dir = arena.run_arena("ko-stub", ["s@low"], episodes=3, max_turns=3,
                              effort="low", seed_base=100, run_root=tmp_path,
                              repeat_seed=True)

    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["seeds"] == [100, 100, 100]             # 전판 동일 시드

    events = [json.loads(l) for l in
              (run_dir / "models" / "s@low" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    ends = [e for e in events if e["type"] == "episode_end"]
    assert len(ends) == 3
    assert {e["target"] for e in ends} == {"0"}        # seed 100 → 정답 "0", 전판 동일

    # summary 집계와 verify_run 재생 모두 중복 seeds에서 무해.
    summary = json.loads((run_dir / "models" / "s@low" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert len(summary["episodes"]) == 3
    assert "usage" in summary
    assert arena.verify_run(run_dir)["ok"] is True


def test_repeat_seed_measurement_key_distinct():
    def mk(seeds):
        return arena.measurement_key({"game": "g", "game_version": "1",
                                      "oracle": {}, "seeds": seeds, "max_turns": 3})
    k_repeat = mk([100, 100])       # 반복
    k_seq = mk([100, 101])          # 순차(다른 문제)
    k_single = mk([100])            # 단판
    assert k_repeat != k_seq
    assert k_repeat != k_single
    assert k_seq != k_single


def test_repeat_seed_reuse_isolation(monkeypatch, tmp_path):
    game = StubGame(max_turns=3)
    monkeypatch.setattr(arena, "build_game", lambda *a, **k: game)
    monkeypatch.setattr(embed, "available", lambda: False)
    _seq_stamps(monkeypatch)
    calls = {"n": 0}
    monkeypatch.setattr(client, "call", _counting(calls, _stub_bot))

    # 단판 [100] 완주 런.
    arena.run_arena("ko-stub", ["s@low"], episodes=1, max_turns=3, effort="low",
                    seed_base=100, run_root=tmp_path)

    # 반복 [100,100] 첫 발사: 키가 달라 단판 완주가 있어도 미적중 → 재측정.
    calls["n"] = 0
    r_rep1 = arena.run_arena("ko-stub", ["s@low"], episodes=2, max_turns=3, effort="low",
                             seed_base=100, run_root=tmp_path, repeat_seed=True)
    assert calls["n"] > 0
    man1 = json.loads((r_rep1 / "manifest.json").read_text(encoding="utf-8"))
    assert "reused_from" not in man1["participants"][0]

    # 같은 반복 조건 재발사: 재사용 적중.
    calls["n"] = 0
    r_rep2 = arena.run_arena("ko-stub", ["s@low"], episodes=2, max_turns=3, effort="low",
                             seed_base=100, run_root=tmp_path, repeat_seed=True)
    assert calls["n"] == 0
    man2 = json.loads((r_rep2 / "manifest.json").read_text(encoding="utf-8"))
    assert man2["participants"][0]["reused_from"] == r_rep1.name


# ----------------------------------------------------------------------
# 오라클 OOV 병목 수정 — 메모이즈 + in-flight 디둡 + 단어별 논블로킹
# (실런: OOV 첫 추측 "사람"을 16레인이 전역 락 뒤에 직렬로 줄 서 6분+ 동반 정지)
# ----------------------------------------------------------------------
def _stub_oracle_embed(monkeypatch, embed_fn):
    """네트워크 없이 EmbeddingOracle을 만들 수 있게 어휘 로딩/모델정보를 스텁하고,
    OOV 임베딩 경로(embed.embed)만 embed_fn으로 대체한다."""
    monkeypatch.setattr(sm.embed, "model_info", lambda m: {"digest": "x"})
    monkeypatch.setattr(sm.embed, "embed_vocab_cached",
                        lambda words, prefix=True, *, model=None: [[1.0, 0.0]] * len(words))
    monkeypatch.setattr(sm.embed, "embed", embed_fn)


def test_oracle_oov_vector_memoized(monkeypatch):
    # 어휘 밖 단어를 두 번 평가해도 embed는 1회만(프로세스 수명 메모이즈, 순수 메모이제이션).
    calls = {"n": 0}

    def fake_embed(texts, prefix=True, *, model=None):
        calls["n"] += 1
        return [[0.3, 0.4] for _ in texts]

    _stub_oracle_embed(monkeypatch, fake_embed)
    oracle = sm.EmbeddingOracle(words=("가", "나"))
    prepared = oracle.prepare("가")          # target은 어휘 안 → embed 호출 없음
    oracle.evaluate(prepared, "사람")         # OOV 첫 평가 → embed 1회
    oracle.evaluate(prepared, "사람")         # 메모이즈 히트 → 추가 호출 없음
    assert calls["n"] == 1


def test_oracle_same_oov_word_concurrent_single_call(monkeypatch):
    # 여러 레인이 동시에 같은 OOV 단어를 추측해도 HTTP는 1회만(in-flight 공유).
    calls = {"n": 0}
    lock = threading.Lock()

    def slow_embed(texts, prefix=True, *, model=None):
        with lock:
            calls["n"] += 1
        time.sleep(0.3)
        return [[0.3, 0.4] for _ in texts]

    _stub_oracle_embed(monkeypatch, slow_embed)
    oracle = sm.EmbeddingOracle(words=("가", "나"))
    prepared = oracle.prepare("가")
    threads = [threading.Thread(target=lambda: oracle.evaluate(prepared, "사람"))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1


def test_oracle_different_oov_words_embed_in_parallel(monkeypatch):
    # 서로 다른 OOV 단어는 서로를 막지 않는다(단어별 논블로킹) → 총 소요 ≈ 단건.
    def slow_embed(texts, prefix=True, *, model=None):
        time.sleep(0.5)
        return [[0.3, 0.4] for _ in texts]

    _stub_oracle_embed(monkeypatch, slow_embed)
    oracle = sm.EmbeddingOracle(words=("가", "나"))
    prepared = oracle.prepare("가")
    done = {}

    def worker(w):
        oracle.evaluate(prepared, w)
        done[w] = True

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(w,)) for w in ("사람", "사랑")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    assert done == {"사람": True, "사랑": True}
    assert elapsed < 0.9            # 병렬 ~0.5s (직렬이면 ~1.0s), 여유 마진


def test_oracle_embed_failure_message_names_cause(monkeypatch):
    # 재시도 소진(embed 예외) → 레인 실패 메시지에 원인·모델명이 드러난다.
    import pytest

    def boom_embed(texts, prefix=True, *, model=None):
        raise socket.timeout("timed out")

    _stub_oracle_embed(monkeypatch, boom_embed)
    oracle = sm.EmbeddingOracle(words=("가", "나"))   # 기본 모델 = ORACLE_MODEL
    prepared = oracle.prepare("가")
    with pytest.raises(RuntimeError) as ei:
        oracle.evaluate(prepared, "사람")
    msg = str(ei.value)
    assert "오라클 임베딩 실패" in msg
    assert sm.ORACLE_MODEL in msg                     # 미로드 의심 모델명(honcho 태그)


def test_manifest_includes_pid(monkeypatch, tmp_path):
    # 웹 정지 기능의 killpg 대상 — manifest에 정수 pid(런 프로세스).
    game = KoreanSemantle(FakeOracle(), max_turns=1)
    _patch_engine(monkeypatch, game)
    run_dir = arena.run_arena("ko-semantle", ["m@low"], episodes=1, max_turns=1,
                              seed_base=1, run_root=tmp_path)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(man["pid"], int)
    assert man["pid"] == os.getpid()   # run_arena는 동일 프로세스(스레드)에서 돈다


# ----------------------------------------------------------------------
# 시작_기록 — 에피소드별 꼴찌권 랜덤 단어의 실채점(레인별 의미 다양화 + 공정성)
# (숫자 노이즈 time은 무효였고, 레인마다 "의미로 읽는" 텍스트가 달라야 고착이 풀린다)
# ----------------------------------------------------------------------
class _BandOracle:
    """REFERENCE_WORDS 전체를 해시 좌표 거리로 결정론 채점 — band 멤버십 검증용."""

    words = sm.REFERENCE_WORDS
    model = "band-stub"

    @property
    def metadata(self):
        return {"embedding_model": self.model, "reference_words": len(self.words),
                "vocab_digest": "sha256:band"}

    @staticmethod
    def _c(w):
        return int(hashlib.sha256(w.encode()).hexdigest()[:12], 16)

    def prepare(self, target):
        return {"target": target}

    def evaluate(self, prepared, guess):
        ref = prepared["target"]
        if guess == ref:
            return SimilarityFeedback(1.0, 1)
        d = abs(self._c(ref) - self._c(guess))
        rank = 1 + sum(1 for w in self.words if abs(self._c(ref) - self._c(w)) < d)
        return SimilarityFeedback(1.0 / (1.0 + rank), rank)

    def pair_cosine(self, a, b):
        return 1.0 if a == b else 0.0


def test_start_record_deterministic_and_varies_by_nonce():
    game = KoreanSemantle(_BandOracle(), max_turns=40)
    a = game.reset(1, nonce="1000").private["start_record"]
    b = game.reset(1, nonce="1000").private["start_record"]
    assert a == b                                   # 같은 seed+nonce → 동일(결정론)
    words = {game.reset(1, nonce=str(x)).private["start_record"]["단어"] for x in range(30)}
    assert len(words) >= 5                           # 다른 nonce → 대체로 다른 단어


def test_start_record_is_bottom_band_and_not_target():
    game = KoreanSemantle(_BandOracle(), max_turns=40)
    for nonce in ("1", "2", "3", "7", "42"):
        state = game.reset(1, nonce=nonce)
        prep, target = state.private["oracle"], state.secret
        graded = sorted(
            ((game.oracle.evaluate(prep, w).similarity, w)
             for w in sm.REFERENCE_WORDS if w != target),
            key=lambda sw: (sw[0], sw[1]))
        band = {w for _, w in graded[:sm._START_BAND_K]}
        sr = state.private["start_record"]
        assert sr["단어"] in band                    # 유사도 최하위 K밴드
        assert sr["단어"] != target                  # 정답 아님
        # 실채점 일치: 시작_기록 순위·유사도가 evaluate 결과와 같다
        fb = game.oracle.evaluate(prep, sr["단어"])
        assert sr["순위"] == fb.rank
        assert sr["유사도"] == round(fb.similarity * 100, 2)


def test_render_includes_start_record_static_block():
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    state = game.reset(1)
    payload = json.loads(game.render(state))
    sr = payload["시작_기록"]
    assert set(sr) == {"단어", "유사도", "순위"}       # {단어,유사도,순위} 포맷
    assert isinstance(sr["단어"], str) and isinstance(sr["순위"], int)
    # 정적(불변) 블록에 위치: time 뒤, 이전_기록(append-only) 앞 → 프리픽스 캐시 정렬
    blob = game.render(state)
    assert blob.index('"time"') < blob.index('"시작_기록"') < blob.index('"이전_기록"')
    assert any("시작_기록" in r for r in payload["규칙"])   # 설명 규칙 한 줄
    # 에피소드 내 불변(턴이 지나도 동일)
    game.step(state, game.parse(_guess("의사")))
    assert json.loads(game.render(state))["시작_기록"] == sr


def test_start_record_does_not_affect_best_rank_or_history():
    game = KoreanSemantle(FakeOracle(), max_turns=5)
    state = game.reset(1)
    p0 = json.loads(game.render(state))
    assert p0["최고_순위"] is None                    # 추측 전 → 시작_기록 있어도 null
    assert p0["이전_기록"] == []                       # 시작_기록은 이전_기록 오염 안 함
    game.step(state, game.parse(_guess("의사")))        # 실제 추측 rank 2
    p1 = json.loads(game.render(state))
    assert p1["최고_순위"] == {"순위": 2, "상위백분위": 50}   # 모델 추측 기준만
    res = game.result(state)
    assert res["turns"] == 1                           # 시작_기록은 턴 미소모
    assert res["best_rank"] == 2


def test_verify_rederives_start_record_from_nonce(monkeypatch, tmp_path):
    game = KoreanSemantle(FakeOracle(), max_turns=3)
    _patch_engine(monkeypatch, game)
    run_dir = arena.run_arena("ko-semantle", ["m@low"], episodes=2, max_turns=3,
                              seed_base=7, run_root=tmp_path)
    events = [json.loads(l) for l in
              (run_dir / "models" / "m@low" / "events.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    ends = [e for e in events if e["type"] == "episode_end"]
    assert ends and all(set(e["시작_기록"]) == {"단어", "유사도", "순위"} for e in ends)
    # verify가 seed+nonce로 시작_기록을 재도출해 일치 → ok
    assert arena.verify_run(run_dir)["ok"] is True
    # 저장된 시작_기록을 위조하면 verify가 잡는다(재도출 불일치)
    ev_path = run_dir / "models" / "m@low" / "events.jsonl"
    tampered = []
    for line in ev_path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("type") == "episode_end":
            e["시작_기록"] = {"단어": "가짜", "유사도": 0.0, "순위": 999}
        tampered.append(json.dumps(e, ensure_ascii=False))
    ev_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")
    assert arena.verify_run(run_dir)["ok"] is False
