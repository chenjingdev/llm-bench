"""bench.arena_web API 계약 테스트 (v2 — 참가자 = 모델×effort).

커버:
  - /api/index, /api/run/<id>, /api/model/<m> 스키마 (slug/participants)
  - v2(slug 디렉토리·effort) + 레거시(모델 id 디렉토리·effort 없음) 둘 다 서빙
  - /api/run/.../events?after=N 증분, slug 세그먼트(@) 통과
  - 경로 방어(트래버설 · 잘못된 세그먼트 · 없는 run → 404) — @ 허용하되 .. 차단
  - target 필터(진행 중 에피소드 은닉 · 끝난 에피소드 공개)
  - POST /api/run 검증(participants → slug argv, --effort 없음; 각종 불량 400)
  - 같은 모델 다른 effort = 두 참가자
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request

import pytest

from bench import arena_web, config
from tests.arena_fixture import build


# --- HTTP 헬퍼 ----------------------------------------------------------
def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, body


def _get_raw(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return r.status, r.read().decode("utf-8"), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), ""


def _post(base, path, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, body


# --- 서버 픽스처 --------------------------------------------------------
class _Srv:
    def __init__(self, base, root, spawns):
        self.base, self.root, self.spawns = base, root, spawns


@pytest.fixture()
def server(tmp_path, monkeypatch):
    root = build(tmp_path / "arena")
    spawns = []
    monkeypatch.setattr(arena_web, "spawn_run", lambda argv: (spawns.append(argv) or 4242))
    srv = arena_web.make_server("127.0.0.1", 0, root)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield _Srv(f"http://127.0.0.1:{port}", root, spawns)
    finally:
        srv.shutdown()
        srv.server_close()


LIVE = "arena-fixture-live"          # v2: participants, dir=slug
DONE = "arena-fixture-done"          # 레거시: dir=모델 id
HK_HIGH = "claude-haiku-4-5@high"
HK_LOW = "claude-haiku-4-5@low"


# --- 인덱스 / 페이지 ----------------------------------------------------
def test_index_schema(server):
    status, data = _get(server.base, "/api/index")
    assert status == 200
    ids = [r["run_id"] for r in data["runs"]]
    assert LIVE in ids and DONE in ids
    assert data["runs"][0]["run_id"] == LIVE
    # v2 런의 models는 slug 리스트(@ 포함)
    assert any("@" in m for m in data["runs"][0]["models"])


def test_index_html_served(server):
    status, html, ctype = _get_raw(server.base, "/")
    assert status == 200
    assert "text/html" in ctype
    assert "관전 콘솔" in html and "MATCH" in html
    assert "claude-haiku-4-5" in html          # 카탈로그 주입
    assert "__ARENA_CONFIG__" not in html       # 플레이스홀더 치환됨


def test_trend_panel_in_html():
    # 하단 '접근 궤적' 비교 패널 + 축 의미 라벨이 서버 렌더 HTML에 정적으로 존재.
    html = arena_web._render_index()
    assert 'id="trend"' in html
    assert 'id="trendBody"' in html
    assert '접근 궤적' in html
    assert '위=정답' in html


# --- run 상세: v2 + 레거시 ----------------------------------------------
def test_run_schema_v2(server):
    status, data = _get(server.base, f"/api/run/{LIVE}")
    assert status == 200
    assert data["manifest"]["episodes"] == 2
    assert isinstance(data["manifest"].get("participants"), list)
    models = data["models"]
    assert len(models) == 5
    # 같은 모델 두 effort → slug 두 개 존재
    assert HK_HIGH in models and HK_LOW in models
    m = models[HK_HIGH]
    assert set(m) == {"live", "summary", "events_count"}
    assert m["live"]["model"] == "claude-haiku-4-5"   # 순수 id
    assert m["live"]["effort"] == "high"              # v2 live에 effort
    assert m["summary"]["effort"] == "high"


def test_run_schema_legacy(server):
    status, data = _get(server.base, f"/api/run/{DONE}")
    assert status == 200
    assert "participants" not in data["manifest"]     # 레거시엔 없음
    models = data["models"]
    assert "claude-haiku-4-5" in models                # dir=모델 id
    assert all("@" not in k for k in models)
    live = models["claude-haiku-4-5"]["live"]
    assert live["model"] == "claude-haiku-4-5"
    assert "effort" not in live                        # 레거시 live엔 effort 없음(파생은 프론트)


def test_run_unknown_404(server):
    assert _get(server.base, "/api/run/does-not-exist")[0] == 404


def test_run_live_never_has_target(server):
    _, data = _get(server.base, f"/api/run/{LIVE}")
    for m in data["models"].values():
        assert "target" not in m["live"]


# --- events 증분 + slug(@) 세그먼트 -------------------------------------
def test_events_slug_segment_and_increment(server):
    path = f"/api/run/{LIVE}/model/{HK_HIGH}/events"   # @ 세그먼트가 통과해야 함
    status, full = _get(server.base, path + "?after=0")
    assert status == 200
    n = full["count"]
    assert n == len(full["events"]) and n > 0
    _, inc = _get(server.base, path + f"?after={n}")
    assert inc["events"] == [] and inc["count"] == n
    _, mid = _get(server.base, path + "?after=2")
    assert len(mid["events"]) == n - 2 and mid["after"] == 2


# --- target 필터 --------------------------------------------------------
def test_target_hidden_for_inprogress_episode(server):
    _, run = _get(server.base, f"/api/run/{LIVE}")
    for slug in run["models"]:
        _, data = _get(server.base, f"/api/run/{LIVE}/model/{slug}/events?after=0")
        for ev in data["events"]:
            if ev.get("episode") == 2:
                assert "target" not in ev, (slug, ev)
            assert ev.get("target") != "바다", (slug, ev)   # ep2 정답 절대 누수 금지


def test_target_shown_for_finished_episode(server):
    _, data = _get(server.base, f"/api/run/{LIVE}/model/claude-sonnet-5@medium/events?after=0")
    ends = [e for e in data["events"] if e.get("type") == "episode_end" and e.get("episode") == 1]
    assert ends and ends[0].get("target") == "사진"


# --- 경로 방어 ----------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "/api/run/..%2f..%2fetc",
    "/api/run/..",
    "/api/run/%2e%2e",
    "/api/run/foo%2Fbar",
    "/api/run/arena-fixture-live/model/..%2f..%2f..%2fpasswd/events",
    "/api/run/arena-fixture-live/model/foo%2Fbar/events",
    "/api/model/..%2f..%2fetc",
    "/api/model/claude-haiku-4-5@high",     # @는 순수 모델 id가 아니므로 model 조회는 404
])
def test_path_traversal_blocked(server, bad):
    assert _get(server.base, bad)[0] == 404


def test_at_segment_allowed_but_scoped(server):
    # @ slug 세그먼트는 허용(정상 200), 존재하지 않는 run/slug는 root 밖이 아니라 빈 목록/404
    assert _get(server.base, f"/api/run/{LIVE}/model/{HK_LOW}/events?after=0")[0] == 200


def test_events_encoded_at_slug(server):
    # 프론트는 encodeURIComponent(slug)로 @를 %40으로 보낸다 → 서버가 디코드해 처리해야 함
    status, data = _get(server.base, f"/api/run/{LIVE}/model/claude-haiku-4-5%40high/events?after=0")
    assert status == 200 and data["count"] > 0


# --- 모델 단위 조회(참가자/effort) --------------------------------------
def test_model_endpoint_participants(server):
    status, data = _get(server.base, "/api/model/claude-haiku-4-5")
    assert status == 200
    assert data["model"] == "claude-haiku-4-5"
    by_run = {r["run_id"]: r for r in data["runs"]}
    assert set(by_run) == {LIVE, DONE}
    # v2 런: 같은 모델 두 effort → 참가자 2, effort {high, low}
    live_efs = sorted(p["effort"] for p in by_run[LIVE]["participants"])
    assert live_efs == ["high", "low"]
    # 레거시 런: 참가자 1, effort는 run.effort에서 유도
    dparts = by_run[DONE]["participants"]
    assert len(dparts) == 1 and dparts[0]["slug"] == "claude-haiku-4-5"
    assert dparts[0]["effort"] == "low" and dparts[0]["summary"] is not None
    # 참여 안 한 모델
    _, none = _get(server.base, "/api/model/claude-opus-4-0")
    assert none["runs"] == []


# --- POST /api/run 검증 (v2) --------------------------------------------
def test_post_participants_spawns(server):
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"},
                             {"model": "claude-haiku-4-5", "effort": "high"},
                             {"model": "codex-5.6-luna", "effort": "medium"}],
            "episodes": 2, "max_turns": 15}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202
    assert data["ok"] is True and data["pid"] == 4242
    assert data["slugs"] == ["claude-haiku-4-5@low", "claude-haiku-4-5@high", "codex-5.6-luna@medium"]
    argv = server.spawns[0]
    assert argv[0] == sys.executable
    assert argv[1:6] == ["-m", "bench", "arena", "run", "--game"]
    assert "--effort" not in argv                     # effort는 slug에 내장
    mi = argv.index("--models")
    assert argv[mi + 1:mi + 4] == ["claude-haiku-4-5@low", "claude-haiku-4-5@high", "codex-5.6-luna@medium"]
    assert argv[argv.index("--episodes") + 1] == "2"
    assert argv[argv.index("--max-turns") + 1] == "15"


def test_post_legacy_body_converted(server):
    # participants 없이 models+effort → participants로 관용 변환
    status, data = _post(server.base, "/api/run",
                         {"models": ["claude-haiku-4-5", "codex-5.6-luna"],
                          "effort": "high", "episodes": 1, "max_turns": 5})
    assert status == 202
    assert data["slugs"] == ["claude-haiku-4-5@high", "codex-5.6-luna@high"]


def test_post_participant_cap_boundary(server):
    catalog = list(config.MODEL_ALIASES)
    efforts = list(arena_web.EFFORTS)
    combos = [{"model": m, "effort": e} for m in catalog for e in efforts]
    assert len(combos) >= 33
    # 32 = 상한 통과
    ok, _ = _post(server.base, "/api/run",
                  {"participants": combos[:32], "episodes": 1, "max_turns": 5})
    assert ok == 202
    # 33 = 초과 거부
    server.spawns.clear()
    bad, data = _post(server.base, "/api/run",
                      {"participants": combos[:33], "episodes": 1, "max_turns": 5})
    assert bad == 400 and "participants" in data["error"]
    assert server.spawns == []


@pytest.mark.parametrize("body,frag", [
    ({"participants": [], "episodes": 1, "max_turns": 5}, "participants"),
    ({"participants": [{"model": "nope", "effort": "low"}], "episodes": 1, "max_turns": 5}, "카탈로그"),
    ({"participants": [{"model": "bad/seg", "effort": "low"}], "episodes": 1, "max_turns": 5}, "형식"),
    ({"participants": [{"model": "claude-haiku-4-5", "effort": "ultra"}], "episodes": 1, "max_turns": 5}, "effort"),
    ({"participants": [{"model": "claude-haiku-4-5", "effort": "low"},
                       {"model": "claude-haiku-4-5", "effort": "low"}], "episodes": 1, "max_turns": 5}, "중복"),
    ({"participants": [{"model": "claude-haiku-4-5", "effort": "low"}], "episodes": 0, "max_turns": 5}, "episodes"),
    ({"participants": [{"model": "claude-haiku-4-5", "effort": "low"}], "episodes": 1, "max_turns": 0}, "max_turns"),
    ({"participants": [{"model": "claude-haiku-4-5", "effort": "low"}], "episodes": 1, "max_turns": 5, "game": "chess"}, "게임"),
    ({"participants": ["notdict"], "episodes": 1, "max_turns": 5}, "형식"),
])
def test_post_invalid_bodies(server, body, frag):
    status, data = _post(server.base, "/api/run", body)
    assert status == 400
    assert frag in data["error"]
    assert server.spawns == []


def test_post_same_model_two_efforts_ok(server):
    # 같은 모델 다른 effort는 중복이 아니라 두 참가자
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "claude-haiku-4-5", "effort": "low"},
                                           {"model": "claude-haiku-4-5", "effort": "high"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 202 and len(data["slugs"]) == 2


def test_post_codex_max_rejected(server):
    # codex CLI엔 max가 없다 → 미지원 effort는 400(플라시보 레인 차단).
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "codex-5.6-luna", "effort": "max"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 400 and "effort" in data["error"]
    assert server.spawns == []


def test_post_unsupported_effort_rejected(server):
    # flash는 low/medium/high만 → xhigh 미지원 → 400(플라시보 레인 차단).
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "gemini-3.5-flash", "effort": "xhigh"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 400 and "effort" in data["error"]
    assert server.spawns == []
    # gpt-oss-120b는 medium만 → low 미지원 → 400.
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "gpt-oss-120b", "effort": "low"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 400 and "effort" in data["error"]
    assert server.spawns == []


def test_post_new_valid_efforts_ok(server):
    # gemini-3-pro는 이제 low도 지원 → 정상 스폰.
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "gemini-3-pro", "effort": "low"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 202
    assert data["slugs"] == ["gemini-3-pro@low"]
    # flash는 medium 지원 → 정상 스폰.
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "gemini-3.5-flash", "effort": "medium"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 202
    assert data["slugs"] == ["gemini-3.5-flash@medium"]


def test_post_claude_max_ok(server):
    # claude는 max까지 지원 → 정상 스폰.
    status, data = _post(server.base, "/api/run",
                         {"participants": [{"model": "claude-haiku-4-5", "effort": "max"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 202
    assert data["slugs"] == ["claude-haiku-4-5@max"]


def test_post_bad_json_400(server):
    assert _post(server.base, "/api/run", None, raw=b"{not json")[0] == 400
    assert server.spawns == []


def test_post_empty_body_400(server):
    assert _post(server.base, "/api/run", None, raw=b"")[0] == 400


def test_post_wrong_path_404(server):
    assert _post(server.base, "/api/nope", {"x": 1})[0] == 404
    assert server.spawns == []


# --- 단위: 순수 함수 ----------------------------------------------------
def test_redact_targets_unit():
    recs = [
        {"type": "turn", "episode": 2, "guess": "x"},
        {"type": "episode_end", "episode": 1, "target": "사진"},
        {"type": "episode_end", "episode": 2, "target": "바다"},
    ]
    out = arena_web._redact_targets(recs, {1})
    assert out[1]["target"] == "사진"
    assert "target" not in out[2]


def test_family_derivation():
    # 카탈로그 id 패턴으로 패밀리·버전 라벨 유도(프론트 하드코딩 아님)
    assert arena_web._family("claude-opus-4-8") == ("Opus", "4.8")
    assert arena_web._family("claude-haiku-4-5") == ("Haiku", "4.5")
    assert arena_web._family("claude-sonnet-5") == ("Sonnet", "5")
    assert arena_web._family("codex-5.6-luna") == ("GPT (Codex)", "5.6 Luna")
    assert arena_web._family("gemini-3-pro") == ("Gemini", "3 Pro")
    # 카탈로그 확장분(codex/gemini/gpt-oss/fable)
    assert arena_web._family("codex-5.6-sol") == ("GPT (Codex)", "5.6 Sol")
    assert arena_web._family("codex-5.4-mini") == ("GPT (Codex)", "5.4 Mini")
    assert arena_web._family("codex-5.6-terra") == ("GPT (Codex)", "5.6 Terra")
    assert arena_web._family("gemini-3.5-flash") == ("Gemini", "3.5 Flash")
    assert arena_web._family("gpt-oss-120b") == ("GPT-OSS", "120B")
    assert arena_web._family("claude-fable-5") == ("Fable", "5")
    # 새 모델도 자동 분류
    assert arena_web._family("claude-opus-9-9") == ("Opus", "9.9")


def test_client_config_families():
    cfg = arena_web._client_config()
    fams = {f["name"]: f for f in cfg["families"]}
    # 선언 순서 보존 + 그룹핑
    assert [f["name"] for f in cfg["families"]][:2] == ["Opus", "Haiku"]
    assert len(fams["Opus"]["models"]) == 6           # 4.0~4.8
    assert len(fams["Haiku"]["models"]) == 1
    assert {m["id"] for f in cfg["families"] for m in f["models"]} == set(config.MODEL_ALIASES)
    assert cfg["maxParticipants"] == 32


def test_client_config_per_model_efforts():
    # 참가자 카탈로그가 모델별 실제 지원 effort를 실어야 한다(플라시보 칩 제거의 단일 소스).
    cfg = arena_web._client_config()
    by_id = {m["id"]: m for m in cfg["models"]}
    assert by_id["gemini-3-pro"]["efforts"] == ["low", "high"]
    assert by_id["gemini-3.5-flash"]["efforts"] == ["low", "medium", "high"]
    assert by_id["gpt-oss-120b"]["efforts"] == ["medium"]
    assert by_id["codex-5.6-luna"]["efforts"] == ["low", "medium", "high", "xhigh"]
    assert by_id["claude-haiku-4-5"]["efforts"] == ["low", "medium", "high", "xhigh", "max"]
    # 패밀리 트리의 모델 dict도 동일하게 effort를 실어야 한다.
    gem = next(m for f in cfg["families"] if f["name"] == "Gemini"
               for m in f["models"] if m["id"] == "gemini-3-pro")
    assert gem["efforts"] == ["low", "high"]


def test_single_effort_model_one_chip_in_html():
    # 단일 effort 모델(gpt-oss-120b → medium)은 CFG에 칩 하나만 실어 카드에 칩 하나만 그려짐.
    html = arena_web._render_index()
    assert '"efforts": ["medium"]' in html
    # 2칩 모델(gemini-3-pro → low/high)도 렌더 CFG에 그대로 실린다.
    assert '"efforts": ["low", "high"]' in html


def test_validate_body_argv_shape():
    payload, err = arena_web._validate_run_body(
        {"participants": [{"model": "claude-haiku-4-5", "effort": "high"}],
         "episodes": 1, "max_turns": 5})
    assert err is None
    assert payload["argv"][1:5] == ["-m", "bench", "arena", "run"]
    assert payload["slugs"] == ["claude-haiku-4-5@high"]
    assert "--effort" not in payload["argv"]
    assert "--models" in payload["argv"]


def test_client_config_carries_full_name():
    cfg = arena_web._client_config()
    by_id = {m["id"]: m for m in cfg["models"]}
    assert by_id["claude-opus-4-8"]["name"] == "Claude Opus 4.8"
    assert by_id["codex-5.6-sol"]["name"] == "GPT-5.6 Sol"
    assert by_id["gemini-3.5-flash"]["name"] == "Gemini 3.5 Flash"
    assert by_id["gpt-oss-120b"]["name"] == "GPT-OSS 120B"


def test_full_names_in_rendered_index():
    html = arena_web._render_index()
    assert "Claude Opus 4.8" in html
    assert "GPT-5.6 Sol" in html
