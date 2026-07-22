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

from bench import arena, arena_web, config
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
FAILED = "arena-fixture-failed"      # v2: status=failed, 고아 레인(live phase=running)
HK_HIGH = "claude-haiku-4-5@high"
HK_LOW = "claude-haiku-4-5@low"
WAIT = "claude-opus-4-6@high"        # 대기 중(디렉터리 없는) 참가자 slug

RULELAB = "arena-rulelab"            # ko-rulelab(비밀 규칙 연구소)
MAZE = "arena-maze"                  # ko-maze(숨은 지도 탐험)
MINE = "arena-minefield"             # ko-minefield(의미 지뢰밭)
REUSE = "arena-fixture-reuse"        # 결과 재사용(측정 경제): fresh(running) + reused(done)
TIE = "arena-fixture-tie"            # 정렬 꼬리(동률 tie-break): 전원 rank1·5턴 → 시간·토큰·slug로만 갈림
PREP = "arena-fixture-preparing"     # 게임 준비 중(오라클 로딩): 예비 manifest만, models/ 없음
PREPFAIL = "arena-fixture-prep-failed"  # 준비 중 죽은 런: status failed + error
EXHAUST = "arena-fixture-exhaust"    # 라이브 런 '턴 소진 실패' 강등: codex(best2,phase done)가 진행 중 아래로
SEEDCOV_A = "arena-seedcov-a"        # 시드 777777·ko-semantle·1판×10턴 (opus@low, sonnet@high)
SEEDCOV_B = "arena-seedcov-b"        # 같은 조건 (sonnet@high 겹침, codex-luna@low, codex-sol@low=model_error)
SEEDCOV_C = "arena-seedcov-c"        # 같은 시드·게임 다른 조건(1판×20턴) → 행 분리
SEEDCOV_L = "arena-seedcov-legacy"   # 시드 없는 구형 런 → '시드 기록 없음' 그룹
SEEDCOV_RULE = "arena-seedcov-rule"  # 시드 314159·ko-rulelab·1판×15턴 (opus@low, gemini@high) — 프리필/조건잠금
MERGE_A = "arena-seedcov-mrg-a"      # 시드 909090·ko-semantle·1판×50턴 (opus@low score .6, sonnet@high)
MERGE_B = "arena-seedcov-mrg-b"      # 시드 909090·ko-semantle·4회 반복×50턴 (opus@low score .9, gemini@high)
COV_SEED = 777777
RULE_SEED = 314159
MERGE_SEED = 909090
REPEAT = "arena-fixture-repeat"      # 반복 런 seeds=[555555]*3 (같은 문제 3회 → 보드 '시도' 선택기)
REPEAT_CONSEC = "arena-repeat-consec"  # 연속 시드 seeds=[555555,555556,555557] (반복과 별도 그룹)
REPEAT_SEED = 555555


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
    assert len(models) == 6                            # 5 시작 + 1 대기(디렉터리 없음)
    # 같은 모델 두 effort → slug 두 개 존재
    assert HK_HIGH in models and HK_LOW in models
    m = models[HK_HIGH]
    assert set(m) == {"live", "summary", "events_count"}
    assert m["live"]["model"] == "claude-haiku-4-5"   # 순수 id
    assert m["live"]["effort"] == "high"              # v2 live에 effort
    assert m["summary"]["effort"] == "high"
    # 대기 중 참가자: manifest 합집합으로 존재하되 live/summary는 None(플레이스홀더)
    assert WAIT in models
    assert models[WAIT]["live"] is None
    assert models[WAIT]["summary"] is None


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
        if m["live"] is not None:                     # 대기 중 참가자는 live=None
            assert "target" not in m["live"]


def test_run_includes_queued_participant_placeholder(server):
    # manifest엔 있으나 아직 시작 안 해 디렉터리 없는 참가자 → live=None 플레이스홀더로 노출.
    status, data = _get(server.base, f"/api/run/{LIVE}")
    assert status == 200
    assert WAIT in data["models"]
    assert data["models"][WAIT]["live"] is None
    assert data["models"][WAIT]["summary"] is None
    assert data["models"][WAIT]["events_count"] == 0
    # manifest에는 참가자로 들어있다(합집합 소스).
    assert WAIT in data["manifest"]["models"]
    assert any(p.get("slug") == WAIT for p in data["manifest"]["participants"])
    # 디스크엔 디렉터리가 없다(다른 5개는 존재).
    on_disk = {p.name for p in (server.root / LIVE / "models").iterdir() if p.is_dir()}
    assert WAIT not in on_disk
    assert len(on_disk) == 5


# --- 종료된(status=failed) 런: 서버는 진실 그대로, 해석은 클라이언트 ------
def test_failed_run_served_honestly(server):
    # 실제 사고 재현: 프로세스 전체 종료(embedding 타임아웃)로 manifest는 failed인데
    # 참가자 live.json은 고아(phase=running)로 남았다. 서버는 편집 없이 진실을 보고한다.
    status, data = _get(server.base, f"/api/run/{FAILED}")
    assert status == 200
    assert data["manifest"]["status"] == "failed"
    assert data["manifest"].get("failure")           # 중단 사유 present
    assert data["manifest"]["finished_at"]           # 종료 시각 present
    # 고아 레인: live.json이 phase='running' 그대로 서빙되어야 한다(표시 계층에서 해석).
    phases = {s: (m["live"] or {}).get("phase") for s, m in data["models"].items()}
    assert any(p == "running" for p in phases.values()), phases
    # 미완 에피소드(episode_end 없음)라 target 누수 없음.
    for m in data["models"].values():
        if m["live"] is not None:
            assert "target" not in m["live"]


def test_failed_run_in_index_after_live(server):
    # 실패 런은 index에 존재하되 runs[0]은 여전히 LIVE(연대순 최신이어도 뒤에 붙임).
    status, data = _get(server.base, "/api/index")
    assert status == 200
    ids = [r["run_id"] for r in data["runs"]]
    assert FAILED in ids
    assert data["runs"][0]["run_id"] == LIVE
    frow = next(r for r in data["runs"] if r["run_id"] == FAILED)
    assert frow["status"] == "failed"


# --- 기록 열람: 참가 수 · 1등 · 업적(런별 카드 계약) ---------------------
def _run_row(data, rid):
    return next(r for r in data["runs"] if r["run_id"] == rid)


def test_index_participants_count_and_winner_shape(server):
    status, data = _get(server.base, "/api/index")
    assert status == 200
    # 참가 수: 대기(디렉터리 없는) 참가자까지 전부 카운트
    assert _run_row(data, LIVE)["participants_count"] == 6
    assert _run_row(data, FAILED)["participants_count"] == 2
    assert _run_row(data, DONE)["participants_count"] == 4
    # 정상 런은 winner(dict, 정확 3키), 무득점·미해결 실패 런은 winner=None
    w = _run_row(data, LIVE)["winner"]
    assert isinstance(w, dict) and set(w) == {"model", "effort", "achievement"}
    assert _run_row(data, FAILED)["winner"] is None


def test_index_winner_selection_and_effort(server):
    # DONE(레거시 semantle): opus가 rank1 해결 → 1등. model은 순수 id, effort는 런 effort로 복원.
    status, data = _get(server.base, "/api/index")
    w = _run_row(data, DONE)["winner"]
    assert w["model"] == "claude-opus-4-8"
    assert w["effort"] == "low"
    assert config.model_name("claude-opus-4-8")   # 클라이언트 nameOf가 붙일 풀네임 존재


def test_index_achievement_per_game(server):
    # 게임별 업적 조립(각 1): 승자 summary.episodes → 한국어 한 줄
    status, data = _get(server.base, "/api/index")
    assert _run_row(data, DONE)["winner"]["achievement"] == "정답 · 4턴"          # ko-semantle
    assert _run_row(data, MAZE)["winner"]["achievement"] == "도착 · 12턴"         # ko-maze
    assert _run_row(data, RULELAB)["winner"]["achievement"] == "5/5 적중 · 실험 4회"  # ko-rulelab
    assert _run_row(data, MINE)["winner"]["achievement"] == "정답 · 4턴"          # ko-minefield


def test_index_achievement_unsolved_best_rank(server):
    # 재사용 런: 완주 참가자(gemini)가 best_rank 2 미해결 → "최고 2위"(순위 계열 미해결 분기)
    status, data = _get(server.base, "/api/index")
    w = _run_row(data, REUSE)["winner"]
    assert w["model"] == "gemini-3-pro" and w["achievement"] == "최고 2위"


def test_history_card_no_model_list_and_new_contract_in_html():
    html = arena_web._render_index()
    assert ".map(slugLabel).join" not in html      # 모델명 나열(옛 renderHistRuns) 제거
    assert "participants_count" in html            # 새 계약 필드가 카드에 배선됨
    assert "r.winner" in html
    assert "rc-win" in html and "rc-ach" in html


def test_history_failed_chip_in_html():
    html = arena_web._render_index()
    assert ".st.failed" in html                    # 중단 칩 톤 CSS
    assert "failed?'중단':stopped?'정지됨':'완료'" in html   # 상태 매핑: failed→'중단', stopped→'정지됨'(런별 카드)


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


# --- 라이브 생성 텍스트 스트림 ------------------------------------------
def test_stream_present_matches_fixture(server):
    status, data = _get(server.base, f"/api/run/{LIVE}/model/claude-opus-4-8@low/stream")
    assert status == 200
    assert data["done"] is False
    assert data["turn"] == 5
    assert "바닷물" in data["text"]
    assert data["model"] == "claude-opus-4-8" and data["effort"] == "low"


def test_stream_done_full_text(server):
    status, data = _get(server.base, f"/api/run/{LIVE}/model/claude-sonnet-5@medium/stream")
    assert status == 200 and data["done"] is True
    assert "GUESS" in data["text"]


def test_stream_waiting_no_deltas(server):
    status, data = _get(server.base, f"/api/run/{LIVE}/model/codex-5.6-luna@low/stream")
    assert status == 200 and data["done"] is False and data["text"] == ""


def test_stream_absent_defaults(server):
    # stream.json 없는 running 참가자 → 서버가 정직한 빈-완료 기본값
    status, data = _get(server.base, f"/api/run/{LIVE}/model/{HK_LOW}/stream")
    assert status == 200
    assert data == {"text": "", "done": True}


@pytest.mark.parametrize("bad", [
    "/api/run/arena-fixture-live/model/..%2f..%2f..%2fpasswd/stream",
    "/api/run/arena-fixture-live/model/foo%2Fbar/stream",
])
def test_stream_path_traversal_blocked(server, bad):
    assert _get(server.base, bad)[0] == 404


def test_stream_encoded_at_slug(server):
    # 프론트는 encodeURIComponent(slug) → @가 %40. 서버가 디코드해 처리(200).
    status, data = _get(server.base, f"/api/run/{LIVE}/model/claude-haiku-4-5%40high/stream")
    assert status == 200 and data["done"] is False


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


def test_post_participant_cap_is_total_combos(server):
    # 선택 상한 제거: participants 상한 = 카탈로그 총 조합 수(모델별 지원 effort 합). 32는 더 이상 상한 아님.
    catalog = list(config.MODEL_ALIASES)
    combos = [{"model": m, "effort": e} for m in catalog for e in config.model_efforts(m)]
    total = len(combos)
    assert total >= 40 and total > 32           # 이제 32 초과도 허용
    # 40개(>32) 유효 참가자 → 통과(상한 없음 증명)
    ok40, _ = _post(server.base, "/api/run",
                    {"participants": combos[:40], "episodes": 1, "max_turns": 5})
    assert ok40 == 202
    # 총 조합 수 전체(경계) → 통과
    server.spawns.clear()
    okall, _ = _post(server.base, "/api/run",
                     {"participants": combos, "episodes": 1, "max_turns": 5})
    assert okall == 202
    # 총 조합 수 초과(중복 1개 덧붙여 길이만 초과) → 400, 스폰 없음
    server.spawns.clear()
    bad, data = _post(server.base, "/api/run",
                      {"participants": combos + [combos[0]], "episodes": 1, "max_turns": 5})
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
    # maxParticipants = 카탈로그 총 조합 수(선택 상한 제거 후: 물리적 최대). defaultWorkers 계약 노출.
    assert cfg["maxParticipants"] == sum(len(config.model_efforts(m)) for m in config.MODEL_ALIASES)
    assert cfg["defaultWorkers"] == arena.MAX_WORKERS_DEFAULT


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


def test_lane_status_and_denom_in_html():
    html = arena_web._render_index()
    assert '방금 기록 갱신' in html
    assert '제자리' in html
    assert '1위 = 정답' in html


def test_abort_labels_in_html():
    # 종료 우선 렌더: 새 상태 칩('중단')과 고아 레인 배지('중단됨') 라벨이 서버 렌더
    # HTML에 실제로 실려 나가야 한다(isLive/배지 로직은 JS라 문자열 표면만 검증).
    html = arena_web._render_index()
    assert '중단됨' in html      # 고아 레인 phase 배지
    assert '중단' in html        # 상단 상태 칩(중단) — 부분 문자열이지만 별도 단언으로 명시
    assert '.status.failed' in html   # 중단 상태 CSS 클래스 존재
    assert '.phase.aborted' in html   # 중단됨 배지 CSS 클래스 존재


# ======================================================================
# 멀티게임: 카탈로그 / 렌더 표면 / 런처
# ======================================================================
def test_client_config_game_meta():
    cfg = arena_web._client_config()
    assert cfg["games"] == ["ko-semantle", "ko-rulelab", "ko-maze", "ko-minefield"]
    gm = cfg["gameMeta"]
    assert set(gm) == set(cfg["games"])
    assert gm["ko-semantle"]["kr"] == "꼬맨틀"
    assert gm["ko-rulelab"]["kr"] == "비밀 규칙 연구소"
    assert gm["ko-maze"]["kr"] == "숨은 지도 탐험" and gm["ko-maze"]["max_turns"] == 40
    assert gm["ko-minefield"]["kr"] == "의미 지뢰밭" and gm["ko-minefield"]["max_turns"] == 40
    # semantle 런처 기본은 보존(무변경)
    assert gm["ko-semantle"]["max_turns"] == 15
    for meta in gm.values():
        assert meta["desc"]                          # 한 줄 설명 present


def test_game_names_and_launcher_in_html():
    html = arena_web._render_index()
    # 4게임 한글명 모두 서버 렌더 HTML(주입된 CFG)에 실려 나간다.
    for kr in ("꼬맨틀", "비밀 규칙 연구소", "숨은 지도 탐험", "의미 지뢰밭"):
        assert kr in html, kr
    # 런처 게임 선택 UI + 게임별 렌더 디스패치 코드가 존재.
    assert 'id="gameSeg"' in html and 'id="gameDesc"' in html
    assert "renderBoardG" in html          # 레인 렌더 디스패치
    assert "mazeMiniMap" in html           # maze 미니맵
    assert "renderTrendMaze" in html and "renderTrendRulelab" in html  # 트렌드 디스패치


# ======================================================================
# 멀티게임: 인덱스 / 매니페스트 game
# ======================================================================
def test_index_lists_new_game_runs(server):
    status, data = _get(server.base, "/api/index")
    assert status == 200
    by_id = {r["run_id"]: r for r in data["runs"]}
    assert by_id[RULELAB]["game"] == "ko-rulelab"
    assert by_id[MAZE]["game"] == "ko-maze"
    assert by_id[MINE]["game"] == "ko-minefield"
    # 신규 게임 런을 붙여도 runs[0]은 여전히 LIVE(semantle)
    assert data["runs"][0]["run_id"] == LIVE


@pytest.mark.parametrize("rid,game,n", [
    (RULELAB, "ko-rulelab", 4),      # 3 시작 + 1 대기(gemini 디렉터리 없음)
    (MAZE, "ko-maze", 3),
    (MINE, "ko-minefield", 3),
])
def test_new_game_manifest_and_models(server, rid, game, n):
    status, data = _get(server.base, f"/api/run/{rid}")
    assert status == 200
    assert data["manifest"]["game"] == game
    assert len(data["models"]) == n


# ======================================================================
# ko-maze: live 지표(거리/탐사/충돌) · 목표 은닉/공개
# ======================================================================
def test_maze_live_metrics_and_no_target(server):
    _, data = _get(server.base, f"/api/run/{MAZE}")
    live = data["models"]["claude-sonnet-5@high"]["live"]   # 탐사 중
    assert live["phase"] == "running"
    assert live["dist"] == 8 and live["bumps"] == 1
    assert live["explored"] is not None
    for m in data["models"].values():
        if m["live"] is not None:
            assert "target" not in m["live"]               # 좌표 은닉


def test_maze_target_hidden_inprogress_shown_finished(server):
    # 도착(완료) 참가자만 목표 좌표 공개, 진행 중 참가자는 어디에도 누수 없음.
    _, done = _get(server.base, f"/api/run/{MAZE}/model/claude-opus-4-8@low/events?after=0")
    ends = [e for e in done["events"] if e.get("type") == "episode_end"]
    assert ends and ends[0]["target"] == "6,6"
    _, prog = _get(server.base, f"/api/run/{MAZE}/model/claude-sonnet-5@high/events?after=0")
    assert not any(e.get("type") == "episode_end" for e in prog["events"])
    for e in prog["events"]:
        assert "target" not in e
        assert "6,6" not in json.dumps(e, ensure_ascii=False)


def test_maze_turn_event_shape(server):
    _, d = _get(server.base, f"/api/run/{MAZE}/model/claude-opus-4-8@low/events?after=0")
    mv = next(e for e in d["events"] if e.get("type") == "turn" and e.get("valid"))
    for k in ("move", "ok", "pos", "open", "bearing", "dist"):
        assert k in mv, k
    for k in ("explored", "bumps"):                        # progress 병합
        assert k in mv, k


# ======================================================================
# ko-rulelab: live 지표(실험/답변) · 규칙 은닉/공개 · correct
# ======================================================================
def test_rulelab_live_metrics_and_rule_hidden(server):
    _, data = _get(server.base, f"/api/run/{RULELAB}")
    live = data["models"]["codex-5.6-luna@low"]["live"]    # 실험 중
    assert live["phase"] == "running"
    assert live["experiments"] == 4 and live["answered"] is False
    # 진행 중 참가자: episode_end 없음 → 규칙(target) 은닉
    _, prog = _get(server.base, f"/api/run/{RULELAB}/model/codex-5.6-luna@low/events?after=0")
    for e in prog["events"]:
        assert "target" not in e
        assert "a×2+b" not in json.dumps(e, ensure_ascii=False)


def test_rulelab_answer_and_rule_reveal(server):
    _, d = _get(server.base, f"/api/run/{RULELAB}/model/claude-opus-4-8@high/events?after=0")
    ans = next(e for e in d["events"] if e.get("type") == "turn" and e.get("kind") == "answer")
    assert ans["correct"] == 5 and isinstance(ans["answer"], list)
    end = next(e for e in d["events"] if e.get("type") == "episode_end")
    assert end["target"] == "a×2+b" and end["correct"] == 5
    assert end["experiments"] == 4 and end["duplicate_tests"] == 0
    # 부분 정답 참가자: correct 3, 중복 실험 1
    _, d2 = _get(server.base, f"/api/run/{RULELAB}/model/claude-sonnet-5@medium/events?after=0")
    end2 = next(e for e in d2["events"] if e.get("type") == "episode_end")
    assert end2["correct"] == 3 and end2["duplicate_tests"] == 1


# ======================================================================
# ko-minefield: 목숨 · 지뢰 이벤트 · 지뢰(mines) 은닉/공개
# ======================================================================
def test_minefield_live_lives_and_no_secret(server):
    _, data = _get(server.base, f"/api/run/{MINE}")
    live = data["models"]["codex-5.6-luna@low"]["live"]    # 진행 중
    assert live["lives"] == 3 and live["best_rank"] == 14
    for m in data["models"].values():
        if m["live"] is not None:
            assert "target" not in m["live"] and "mines" not in m["live"]


def test_minefield_boom_turn_shape(server):
    _, d = _get(server.base, f"/api/run/{MINE}/model/claude-sonnet-5@high/events?after=0")
    booms = [e for e in d["events"] if e.get("type") == "turn" and e.get("mine_event") == "boom"]
    assert len(booms) == 3
    b = booms[0]
    assert "rank" not in b and "similarity" not in b       # 폭발 턴은 정보 몰수
    assert b["lives"] == 2 and b["guess"]                   # 목숨·추측은 존재
    # 마지막 폭발로 목숨 0(mined) → episode_end에 lives_left 0
    end = next(e for e in d["events"] if e.get("type") == "episode_end")
    assert end["lives_left"] == 0 and end["booms"] == 3


def test_minefield_mines_hidden_inprogress_shown_finished(server):
    # 진행 중 참가자: 지뢰(mines)·정답(target) 절대 누수 금지.
    _, prog = _get(server.base, f"/api/run/{MINE}/model/codex-5.6-luna@low/events?after=0")
    assert not any(e.get("type") == "episode_end" for e in prog["events"])
    for e in prog["events"]:
        assert "mines" not in e and "target" not in e
        blob = json.dumps(e, ensure_ascii=False)
        assert "사과" not in blob and "전쟁" not in blob and "질병" not in blob
    # 승리(완료) 참가자: episode_end에서만 정답+지뢰 공개.
    _, win = _get(server.base, f"/api/run/{MINE}/model/claude-opus-4-8@low/events?after=0")
    end = next(e for e in win["events"] if e.get("type") == "episode_end")
    assert end["target"] == "사과" and end["mines"] == ["전쟁", "질병"]
    assert end["solved"] is True and end["best_rank"] == 1


def test_redact_mines_unit():
    # _redact_targets가 진행 중 에피소드의 mines도 target과 함께 제거.
    recs = [
        {"type": "episode_end", "episode": 1, "target": "사과", "mines": ["전쟁", "질병"]},
        {"type": "episode_end", "episode": 2, "target": "바다", "mines": ["불", "물"]},
    ]
    out = arena_web._redact_targets(recs, {1})
    assert out[0]["target"] == "사과" and out[0]["mines"] == ["전쟁", "질병"]
    assert "mines" not in out[1] and "target" not in out[1]


# ======================================================================
# POST /api/run: 신규 게임 스폰
# ======================================================================
@pytest.mark.parametrize("game", ["ko-rulelab", "ko-maze", "ko-minefield"])
def test_post_new_games_spawn(server, game):
    body = {"game": game,
            "participants": [{"model": "codex-5.6-luna", "effort": "low"}],
            "episodes": 1, "max_turns": 10}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["ok"] is True
    assert data["game"] == game
    argv = server.spawns[-1]
    gi = argv.index("--game")
    assert argv[gi + 1] == game


def test_post_unknown_game_still_rejected(server):
    status, data = _post(server.base, "/api/run",
                         {"game": "ko-chess",
                          "participants": [{"model": "codex-5.6-luna", "effort": "low"}],
                          "episodes": 1, "max_turns": 5})
    assert status == 400 and "게임" in data["error"]
    assert server.spawns == []


# ======================================================================
# 측정 경제(결과 재사용, 계약 §9 부록 A)
# ======================================================================
def test_client_config_exposes_suite_seed():
    # 정규 문제 세트 고정 seed가 클라이언트 설정에 실린다(엔진 config에서 안전 로드).
    cfg = arena_web._client_config()
    assert cfg["suiteSeed"] == getattr(config, "ARENA_SUITE_SEED_BASE", 314159)
    assert isinstance(cfg["suiteSeed"], int) and not isinstance(cfg["suiteSeed"], bool)


def test_launcher_seed_input_in_html():
    # 런처 시드 입력: #fSeed 숫자 입력칸(기본값 프리필) + 🎲 랜덤 버튼 + 재측정 토글 유지.
    html = arena_web._render_index()
    assert 'id="fSeed"' in html                       # 체크박스 토글 대신 시드 입력칸
    assert 'id="seedDice"' in html and '🎲' in html    # 랜덤 시드 생성 버튼
    assert 'id="fRemeasure"' in html                  # '전부 다시 측정' 토글은 유지
    assert 'id="fSuite"' not in html                  # 옛 정규 세트 토글은 제거
    assert '시드(문제 세트)' in html                   # 라벨
    assert '자동 재활용' in html                       # 재활용 취지 안내 문구
    assert '전부 다시 측정' in html
    assert '"suiteSeed"' in html                      # 기본값 프리필용 CFG seed 주입
    assert 'value="314159"' in html                   # 기본 시드 정적 프리필


def test_launcher_seed_wiring_in_html():
    # 시드 로직: 항상 body.seed 전송(파싱) + 랜덤 주사위 + 부트 프리필 + 유효성 문구.
    html = arena_web._render_index()
    assert "reuse:!remeasure, seed, repeat_seed:true, workers}" in html   # seed·repeat_seed·workers 항상 body에
    assert "Math.random()*0x80000000" in html         # 🎲 → 0~2^31 랜덤 시드
    assert "$('#fSeed').value=CFG.suiteSeed" in html   # 부트 시 CFG 기본값 프리필
    assert "시드는 0 이상의 정수여야 합니다" in html   # 정직한 유효성 안내


def test_topbar_seed_chip_in_html():
    # 상단 칩: manifest.seeds[0]을 라벨 있는 '시드' 칩으로 표기(라벨 없는 숫자 금지).
    html = arena_web._render_index()
    assert "chip('시드',man.seeds[0])" in html
    assert "Array.isArray(man.seeds)" in html          # seeds 부재(구형 런)면 생략


def test_reuse_badge_render_surface_in_html():
    # 재사용 배지: reused_from 참가자에 '재사용' 배지 + 출처 run_id 툴팁(레인/드로어).
    html = arena_web._render_index()
    assert 'reusedBadge' in html
    assert 'rbadge' in html                           # 배지 CSS 클래스
    assert '재사용' in html                            # 배지 라벨


def test_post_suite_seed_and_reuse_default(server):
    # 정규 세트 ON(=seed 전달) + reuse 기본 true → --seed 있고 --no-reuse 없음.
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5, "seed": 314159, "reuse": True}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202
    assert data["seed"] == 314159 and data["reuse"] is True
    argv = server.spawns[0]
    assert argv[argv.index("--seed") + 1] == "314159"
    assert "--no-reuse" not in argv


def test_post_reuse_false_adds_no_reuse(server):
    # '전부 다시 측정'(reuse:false) → --no-reuse. seed 미전달 → --seed 없음.
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5, "reuse": False}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["reuse"] is False and data["seed"] is None
    argv = server.spawns[0]
    assert "--no-reuse" in argv
    assert "--seed" not in argv


def test_post_reuse_defaults_true_when_absent(server):
    # reuse/seed 키 없음 → reuse 기본 True(--no-reuse 없음), seed 없음(기존 argv 형태 보존).
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["reuse"] is True and data["seed"] is None
    argv = server.spawns[0]
    assert "--no-reuse" not in argv and "--seed" not in argv


def test_post_seed_and_no_reuse_together(server):
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5, "seed": 42, "reuse": False}
    status, _ = _post(server.base, "/api/run", body)
    assert status == 202
    argv = server.spawns[0]
    assert argv[argv.index("--seed") + 1] == "42"
    assert "--no-reuse" in argv


@pytest.mark.parametrize("bad", [{"seed": "abc"}, {"seed": True}, {"seed": 3.5}, {"seed": -1}])
def test_post_bad_seed_rejected(server, bad):
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5}
    body.update(bad)
    status, data = _post(server.base, "/api/run", body)
    assert status == 400 and "seed" in data["error"]
    assert server.spawns == []


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_post_bad_reuse_rejected(server, bad):
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5, "reuse": bad}
    status, data = _post(server.base, "/api/run", body)
    assert status == 400 and "reuse" in data["error"]
    assert server.spawns == []


def test_reuse_run_manifest_carries_reused_from(server):
    # 서버는 manifest.participants[i].reused_from · measurement_key를 편집 없이 통과.
    status, data = _get(server.base, f"/api/run/{REUSE}")
    assert status == 200
    man = data["manifest"]
    assert man["measurement_key"]
    parts = {p["slug"]: p for p in man["participants"]}
    assert parts["gemini-3-pro@high"]["reused_from"] == DONE
    assert "reused_from" not in parts["claude-opus-4-8@low"]      # fresh엔 없음
    # 재사용 참가자: 런 시작부터 완료(summary·events 존재). fresh: 진행 중(phase running).
    reused = data["models"]["gemini-3-pro@high"]
    assert reused["summary"] is not None and reused["events_count"] > 0
    assert data["models"]["claude-opus-4-8@low"]["live"]["phase"] == "running"


def test_reuse_run_in_index_after_live(server):
    status, data = _get(server.base, "/api/index")
    assert status == 200
    ids = [r["run_id"] for r in data["runs"]]
    assert REUSE in ids
    assert data["runs"][0]["run_id"] == LIVE                      # runs[0]은 여전히 LIVE


def test_reuse_run_fresh_target_hidden(server):
    # fresh(진행 중): episode_end 없음 → 정답 은닉. reused: 자기 episode_end에서만 공개.
    _, fresh = _get(server.base, f"/api/run/{REUSE}/model/claude-opus-4-8@low/events?after=0")
    assert not any(e.get("type") == "episode_end" for e in fresh["events"])
    for e in fresh["events"]:
        assert "target" not in e
        assert "여름" not in json.dumps(e, ensure_ascii=False)
    _, reused = _get(server.base, f"/api/run/{REUSE}/model/gemini-3-pro@high/events?after=0")
    end = next(e for e in reused["events"] if e.get("type") == "episode_end")
    assert end["target"] == "여름"


# ======================================================================
# 정렬 꼬리(공통 tie-break): 턴 ↑ → 시간(usage.duration_ms) ↑ → 출력 토큰 ↑ → slug
# ======================================================================
from pathlib import Path


def _mk_summary_run(root, rid, parts, game="ko-semantle"):
    """정렬 키만 담은 최소 런을 디스크에 합성한다.

    parts: [{slug, mean_score, median_turns, usage?(dict), solved?}].
    'usage' 키가 없으면 구형(usage 결측) 참가자. (root, run_dict) 반환.
    """
    root = Path(root)
    slugs = [p["slug"] for p in parts]
    manifest = {"run_id": rid, "game": game, "models": slugs,
                "participants": [{"model": p["slug"].split("@")[0],
                                  "effort": p["slug"].split("@", 1)[1] if "@" in p["slug"] else "low",
                                  "slug": p["slug"]} for p in parts]}
    (root / rid).mkdir(parents=True, exist_ok=True)
    (root / rid / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for p in parts:
        model = p["slug"].split("@")[0]
        eff = p["slug"].split("@", 1)[1] if "@" in p["slug"] else "low"
        s = {"model": model, "effort": eff,
             "mean_score": p["mean_score"], "median_turns": p["median_turns"],
             "episodes": [{"type": "episode_end", "episode": 1,
                           "solved": p.get("solved", True), "turns": p["median_turns"],
                           "score": p["mean_score"], "target": "x"}]}
        if "usage" in p:
            s["usage"] = p["usage"]
        d = root / rid / "models" / p["slug"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    run = {"run_id": rid, "game": game, "models": slugs, "effort": None}
    return root, run


def test_run_summary_median_turns_before_usage(tmp_path):
    # 진행도(mean_score) 동률 → 중앙 턴 수가 시간·토큰보다 먼저(적은 쪽 승).
    # 턴이 적은 A는 시간이 더 길어도 이긴다(꼬리 순서 검증).
    root, run = _mk_summary_run(tmp_path, "r-mt", [
        {"slug": "aa-model@low", "mean_score": 0.9, "median_turns": 6,
         "usage": {"duration_ms": 1000, "output_tokens": 10}},
        {"slug": "bb-model@low", "mean_score": 0.9, "median_turns": 4,
         "usage": {"duration_ms": 99000, "output_tokens": 9999}},
    ])
    _, winner = arena_web._run_summary(root, run)
    assert winner["model"] == "bb-model"          # 턴 4 < 6


def test_run_summary_tiebreak_duration_step(tmp_path):
    # mean_score·median_turns 동률 → 소요 시간(usage.duration_ms) 적은 쪽 승.
    root, run = _mk_summary_run(tmp_path, "r-dur", [
        {"slug": "aa-model@low", "mean_score": 0.9, "median_turns": 5,
         "usage": {"duration_ms": 45000, "output_tokens": 500}},   # 토큰 더 적어도
        {"slug": "bb-model@low", "mean_score": 0.9, "median_turns": 5,
         "usage": {"duration_ms": 20000, "output_tokens": 900}},   # 시간이 짧아 승
    ])
    _, winner = arena_web._run_summary(root, run)
    assert winner["model"] == "bb-model"


def test_run_summary_tiebreak_tokens_step(tmp_path):
    # mean_score·median_turns·duration 동률 → 출력 토큰 적은 쪽 승.
    root, run = _mk_summary_run(tmp_path, "r-tok", [
        {"slug": "aa-model@low", "mean_score": 0.9, "median_turns": 5,
         "usage": {"duration_ms": 30000, "output_tokens": 1200}},
        {"slug": "bb-model@low", "mean_score": 0.9, "median_turns": 5,
         "usage": {"duration_ms": 30000, "output_tokens": 700}},   # 토큰 적어 승
    ])
    _, winner = arena_web._run_summary(root, run)
    assert winner["model"] == "bb-model"


def test_run_summary_missing_usage_sorts_last(tmp_path):
    # usage 결측(구형)은 시간 단계에서 무한대 취급 → usage 있는 참가자에게 진다.
    # 단, 그 위 단계(mean_score·median_turns)까지의 우열은 유지.
    root, run = _mk_summary_run(tmp_path, "r-miss", [
        {"slug": "aa-model@low", "mean_score": 0.9, "median_turns": 5},   # usage 없음
        {"slug": "bb-model@low", "mean_score": 0.9, "median_turns": 5,
         "usage": {"duration_ms": 99000, "output_tokens": 9999}},         # 매우 느려도 승
    ])
    _, winner = arena_web._run_summary(root, run)
    assert winner["model"] == "bb-model"
    # 결측이 진행도까지 무너뜨리진 않는다: 점수가 더 높으면 usage 없어도 이긴다.
    root2, run2 = _mk_summary_run(tmp_path, "r-miss2", [
        {"slug": "aa-model@low", "mean_score": 0.95, "median_turns": 9},  # usage 없지만 고득점
        {"slug": "bb-model@low", "mean_score": 0.5, "median_turns": 2,
         "usage": {"duration_ms": 1000, "output_tokens": 10}},
    ])
    _, winner2 = arena_web._run_summary(root2, run2)
    assert winner2["model"] == "aa-model"


def test_index_winner_tiebreak_by_usage(server):
    # 픽스처 tie 런: 4명 전원 rank1·5턴(mean_score·median_turns 동률) → 소요 시간 최소가 1등.
    # sonnet(20s) < codex·opus(45s) < gemini(usage 없음).
    _, data = _get(server.base, "/api/index")
    w = _run_row(data, TIE)["winner"]
    assert w["model"] == "claude-sonnet-5" and w["effort"] == "high"
    assert w["achievement"] == "정답 · 5턴"


def test_tie_run_in_index_after_live(server):
    _, data = _get(server.base, "/api/index")
    ids = [r["run_id"] for r in data["runs"]]
    assert TIE in ids
    assert data["runs"][0]["run_id"] == LIVE                    # runs[0]은 여전히 LIVE


def test_summary_usage_passthrough(server):
    # 서버는 summary.usage(엔진 집계)를 편집 없이 통과시킨다(/api/model).
    _, data = _get(server.base, "/api/model/claude-sonnet-5")
    tie = next(r for r in data["runs"] if r["run_id"] == TIE)
    part = next(p for p in tie["participants"] if p["slug"] == "claude-sonnet-5@high")
    u = part["summary"]["usage"]
    assert u["duration_ms"] == 20000 and u["output_tokens"] == 900


def test_turn_events_carry_usage_and_missing_safe(server):
    # usage 있는 참가자: 턴 이벤트에 usage(duration_ms/output_tokens) 실림.
    _, d = _get(server.base, f"/api/run/{TIE}/model/claude-opus-4-8@high/events?after=0")
    tu = [e for e in d["events"] if e.get("type") == "turn" and "usage" in e]
    assert tu and all(isinstance(e["usage"].get("duration_ms"), int) for e in tu)
    assert tu[0]["usage"]["output_tokens"] == 240
    # usage 없는 구형 참가자(gemini): 턴 이벤트에 usage 키 없음 — 서버가 만들어내지 않는다(결측 안전).
    _, g = _get(server.base, f"/api/run/{TIE}/model/gemini-3-pro@low/events?after=0")
    assert not any("usage" in e for e in g["events"] if e.get("type") == "turn")
    # 결측 참가자도 정상 서빙(에러 없음)되고 이벤트는 존재.
    assert any(e.get("type") == "turn" for e in g["events"])


def test_live_missing_usage_participant_served(server):
    # LIVE 런의 haiku@low는 usage 없는 구형 참가자 — summary에 usage 키 없음, 그래도 정상 서빙.
    _, data = _get(server.base, f"/api/run/{LIVE}")
    assert "usage" not in (data["models"][HK_LOW]["summary"] or {})
    assert "usage" in data["models"]["claude-opus-4-8@low"]["summary"]   # 있는 쪽은 실림


# --- 클라이언트 표면(HTML): 정렬 꼬리 · 레인 자원 지표 · 보드 캡션 ----------
def test_sort_tail_wired_in_html():
    # 공통 꼬리 비교자와 누적 필드가 렌더 JS에 배선됨(JS는 스크린샷으로 판독, 표면만 단언).
    html = arena_web._render_index()
    assert "function tieCmp" in html
    assert "turnsUsed" in html
    assert "durMs" in html and "outTok" in html
    assert "function usageOf" in html                # 이벤트 순회 누적 헬퍼


def test_lane_cost_metric_in_html():
    # 레인 시간·토큰 지표 + 포맷터가 HTML에 존재(라벨 포함).
    html = arena_web._render_index()
    assert "function laneCost" in html
    assert "function fmtDur" in html and "function fmtTok" in html
    assert "costline" in html                        # CSS 클래스
    assert "시간 <b>" in html                         # 시간 라벨
    assert "tok" in html                             # 토큰 단위


def test_lane_cost_usd_in_html():
    # 레인 비용(cost_usd) 표시: usageOf 누적 + viewOf/baseView 전달 + laneCost 칩 + 포맷터.
    html = arena_web._render_index()
    assert "if(typeof u.cost_usd==='number')" in html   # usageOf가 턴 usage의 cost_usd 누적
    assert "costUsd: cost.costUsd" in html               # viewOf(semantle) 전달
    assert "costUsd:cost.costUsd" in html                # baseView(멀티게임) 전달
    assert "function fmtCost" in html
    assert "fmtCost(v.costUsd)" in html                  # laneCost가 비용 칩 렌더
    assert "'cchip cost'" in html                        # 비용 칩 클래스
    # 결측 정직성: 숫자 아니면 null(칩 생략) — 0으로 지어내지 않음. 0은 $0(무과금 정직).
    assert "if(typeof usd!=='number'||!isFinite(usd)) return null;" in html
    assert "if(usd===0) return '$0';" in html


def test_launcher_no_pilot_localstorage_in_html():
    # 프리셋 없음 + localStorage 저장/복원(카탈로그 검증). CFG.pilot 참조 제거(키는 계약상 서버에 유지).
    html = arena_web._render_index()
    assert 'CFG.pilot' not in html                       # 부트 프리셋 참조 제거
    assert 'pilot' in arena_web._client_config()         # 계약 키는 서버에 유지
    assert "const PICKED_KEY='arena.picked.v1'" in html
    assert 'function restorePicked' in html and 'function savePicked' in html
    assert 'let picked=restorePicked();' in html         # 저장값 복원(없으면 빈 Map)
    assert 'savePicked();' in html                       # syncCount(모든 변경 경로)에서 저장
    # 복원 시 카탈로그 검증: 없는 모델·미지원 effort 제외
    body = html.split('function restorePicked', 1)[1].split('function savePicked', 1)[0]
    assert 'CFG.models.find(x=>x.id===pair[0]); if(!mm) return;' in body   # 없는 모델 제외
    assert 'pair[1].filter(e=>sup.includes(e))' in body                    # 미지원 effort 제외


def test_live_nodata_lane_uniform_in_html():
    # 라이브 무데이터 레인 통일: waiting이라도 isLive면 실행 레인과 동일 구조로 렌더 + '첫 턴 진행 중…'.
    html = arena_web._render_index()
    # 4개 렌더러(semantle laneEl + maze/rulelab/mine) 모두 라이브 게이트
    assert html.count("v.phase==='waiting' && !isLive()") >= 4
    # 완료/중단 런의 무데이터 레인만 얇은 waitingLaneG 유지(라이브면 통과)
    assert 'return waitingLaneG(model,idx)' in html
    # 라이브 무데이터 상태 라벨(거짓 지표 금지)
    assert html.count('첫 턴 진행 중') >= 4


def test_lane_compact_heights_in_html():
    # 레인 수직 압축(2-b): 예약 높이·박스 패딩·슬롯 간 gap 축소(레이아웃 시프트 없이 낮춘 행).
    html = arena_web._render_index()
    assert '--lane-min:42px' in html                     # 최소 높이 축소(52→42)
    assert '--lane-pad:3px 18px 3px 18px' in html        # 상하 패딩 축소(8→3)
    assert '.mid-col{display:flex;flex-direction:column;gap:4px' in html  # 슬롯 간 gap 축소(7→4)


def test_board_caption_shows_real_chain_in_html():
    # 보드 캡션이 실제 정렬 사슬을 라벨과 함께 노출(라벨 없는 기준 금지).
    html = arena_web._render_index()
    assert "턴 → 시간 → 토큰 순" in html               # usage 있는 런의 공통 꼬리
    assert "정답 근접 → " in html                      # 꼬맨틀/지뢰밭 진행도 헤드라인
    assert "1위 = 정답" in html                       # 기존 라벨 보존


def test_caption_status_word_and_usage_honesty_in_html():
    # 캡션 선두어가 상태별(실시간/최종/중단 시점)로 갈리고, usage 없는 런은 시간·토큰
    # 순을 약속하지 않는다(정직화). setBoardHead가 isLive·runHasUsage로 분기.
    html = arena_web._render_index()
    assert "'실시간'" in html and "'최종'" in html and "'중단 시점'" in html
    assert "function runHasUsage" in html
    assert "시간·토큰 기록 없는 런" in html            # usage 전무 런의 정직 캡션
    assert "'동률: 턴 → 시간 → 토큰'" in html          # usage 있는 런은 현행 유지


def test_demotion_tier_wired_in_html():
    # 강등 티어: 정답>진행 중>턴 소진 실패>대기. renderBoard(semantle)·cmpFor(멀티) 공통.
    html = arena_web._render_index()
    assert "function sortTier" in html
    assert "if(v.solved) return 0;" in html            # 정답 최상
    assert "if(v.phase==='running') return 1;" in html  # 진행 중
    assert "return 2;" in html                          # 턴 소진·중단 실패
    assert "const ta=sortTier(a.v), tb=sortTier(b.v);" in html   # semantle 정렬 배선
    assert "const tier=(a,b)=>{ const ta=sortTier(a.v),tb=sortTier(b.v);" in html  # cmpFor 배선
    assert ".phase.exhausted" in html                   # 실패 배지 톤
    assert "'턴 소진'" in html                          # 'N턴 소진' 배지 라벨


def test_solve_basis_and_gated_statusline_in_html():
    # 정답 레인 순위 근거('정답까지 N턴') = 정렬 꼬리 turnsUsed와 동일 값 표시.
    # 진행 지표(방금 기록 갱신/제자리)는 라이브에서만(정적 런 부정직 제거).
    html = arena_web._render_index()
    assert "function solveBasis" in html
    assert "정답까지 " in html
    assert "solvebasis" in html                         # 배지 CSS 클래스
    assert "else if(isLive() && v.phase==='running') su=statusLine(v);" in html  # 진행 중 레인만


# ======================================================================
# 라이브 런 '턴 소진 실패' 강등 픽스처(정렬은 클라이언트 JS라 데이터 상태만 서버 검증)
# ======================================================================
def test_exhaust_run_states_served(server):
    # codex는 스레드 종료(live phase='done')·미해결(episode_end 없음) → 강등 대상.
    # opus/sonnet는 진행 중(phase='running'). gemini는 대기(placeholder).
    status, data = _get(server.base, f"/api/run/{EXHAUST}")
    assert status == 200
    assert data["manifest"]["status"] == "running"
    models = data["models"]
    assert models["claude-opus-4-8@low"]["live"]["phase"] == "running"
    assert models["claude-sonnet-5@medium"]["live"]["phase"] == "running"
    cdx = models["codex-5.6-luna@low"]
    assert cdx["live"]["phase"] == "done"              # 스레드 종료(진행 중 아님)
    assert cdx["live"]["best_rank"] == 2               # 진행 중보다 순위는 좋음(그래도 강등)
    # 대기 참가자: 디렉터리 없음 → 플레이스홀더
    assert models["gemini-3-pro@high"]["live"] is None


def test_exhaust_run_no_target_leak(server):
    # 소진 실패 참가자(codex)는 episode_end가 없어 정답이 새지 않는다(진행 중과 같은 ep).
    for slug in ("claude-opus-4-8@low", "claude-sonnet-5@medium", "codex-5.6-luna@low"):
        _, d = _get(server.base, f"/api/run/{EXHAUST}/model/{slug}/events?after=0")
        assert not any(e.get("type") == "episode_end" for e in d["events"]), slug
        for e in d["events"]:
            assert "target" not in e
            assert "노을" not in json.dumps(e, ensure_ascii=False)


def test_exhaust_run_in_index_after_live(server):
    status, data = _get(server.base, "/api/index")
    assert status == 200
    ids = [r["run_id"] for r in data["runs"]]
    assert EXHAUST in ids
    assert data["runs"][0]["run_id"] == LIVE           # runs[0]은 여전히 LIVE
    row = _run_row(data, EXHAUST)
    assert row["status"] == "running" and row["participants_count"] == 4


# ======================================================================
# 게임 준비 중(오라클 로딩) — 예비 manifest 단계
# ======================================================================
def test_preparing_run_served_as_preparing(server):
    # 예비 manifest만 있는 준비 중 런: status=preparing, 오라클/verify 부재, 참가자는
    # 디렉터리 없는 플레이스홀더(live=None). 서버는 편집 없이 진실을 보고한다.
    status, data = _get(server.base, f"/api/run/{PREP}")
    assert status == 200
    man = data["manifest"]
    assert man["status"] == "preparing"
    assert man["finished_at"] is None
    assert "oracle" not in man and "verify" not in man       # 준비 단계엔 부재
    assert "game_version" not in man and "measurement_key" not in man
    assert isinstance(man.get("participants"), list) and man["participants"]
    # models/ 디렉터리가 없으니 전 참가자가 플레이스홀더(live/summary None, events 0)
    assert len(data["models"]) == len(man["participants"])
    for m in data["models"].values():
        assert m["live"] is None and m["summary"] is None and m["events_count"] == 0
    # 디스크에 models/ 디렉터리 자체가 없다(준비 중이라 참가자 미시작).
    assert not (server.root / PREP / "models").exists()


def test_preparing_run_in_index_after_live(server):
    # 준비 중 런은 index에 status=preparing으로 노출, runs[0]은 여전히 LIVE.
    status, data = _get(server.base, "/api/index")
    assert status == 200
    assert data["runs"][0]["run_id"] == LIVE
    prow = _run_row(data, PREP)
    assert prow["status"] == "preparing"
    # 참가 수는 대기 참가자 포함 전부 카운트, 승자는 아직 없음(summary 부재).
    assert prow["participants_count"] == 3
    assert prow["winner"] is None


def test_prep_failed_run_served_honestly(server):
    # 준비 단계에서 죽은 런: 예비 manifest가 status=failed + error로 덮였다.
    # models/ 미도달(디렉터리 없음), finished_at·error 존재. '완료' 아님을 보장.
    status, data = _get(server.base, f"/api/run/{PREPFAIL}")
    assert status == 200
    man = data["manifest"]
    assert man["status"] == "failed"
    assert man.get("error")
    assert man["finished_at"]
    assert "oracle" not in man                                # 준비 중 죽어 오라클 부재
    _, idx = _get(server.base, "/api/index")
    frow = _run_row(idx, PREPFAIL)
    assert frow["status"] == "failed"
    assert frow["winner"] is None


def test_preparing_status_chip_and_islive_in_html():
    # 상단 상태칩: preparing → '준비 중'(전용 톤). isLive가 preparing을 포함해 폴링 유지.
    html = arena_web._render_index()
    assert "'준비 중'" in html or "준비 중" in html
    assert ".status.preparing" in html                        # 전용 칩 톤 CSS
    assert ".st.preparing" in html                            # 런카드 칩 톤 CSS
    # 폴링 유지: isLive 판정에 preparing 포함(문자열 표면 검증).
    assert "if(man.status==='preparing') return true;" in html


def test_preparing_not_mislabeled_done_in_html():
    # 런카드가 preparing을 반드시 먼저 판정 — 아니면 '완료'로 오표기된다(정직성 위반).
    html = arena_web._render_index()
    assert "preparing?'준비 중':running?'LIVE':failed?'중단':stopped?'정지됨':'완료'" in html
    assert "preparing?'preparing':running?'running':failed?'failed':stopped?'stopped':'done'" in html


def test_auto_opening_turn_marker_in_html():
    # 무작위 오프닝(auto:true) 턴을 이력 칩에서 정직히 구분 — 모델 성과처럼 보이면 안 됨.
    html = arena_web._render_index()
    # 칩 클래스에 auto + '배정' 표식 + 안내 title(라벨 없는 구분 금지)
    assert "hchip'+(t.auto?' auto':'')" in html
    assert "el('span','ha','배정')" in html
    assert "시스템 무작위 착수(모델 추측 아님)" in html
    # 히트색(성과 톤)은 auto 아닌 턴에만 — auto는 뮤트 톤 + 점선
    assert ".hchip.auto" in html and ".hchip .ha" in html
    assert "chip.style.background=`rgb(" in html                 # 비-auto 턴은 여전히 히트색


def test_auto_turn_excluded_from_record_emphasis_in_html():
    # '방금 기록 갱신' 등 모델 성과 강조는 모델 실제 턴만 근거 — auto 턴 제외(순위 계산 자체는 무변경).
    html = arena_web._render_index()
    body = html.split("function statusLine(v){", 1)[1].split("function ", 1)[0]
    assert "v.valid.filter(t=>!t.auto)" in body                  # auto 제외한 모델 턴으로 갱신/제자리 판정
    assert "if(!mturns.length) return null;" in body             # auto 착수만 있으면 강조 생략


def test_auto_field_backward_compat_in_html():
    # 하위호환: auto 필드 없는 구 데이터(1.7.0-)는 t.auto가 undefined(falsy)로 취급 →
    #   이력 칩은 히트색 정상 렌더, statusLine 필터도 전 턴 유지(무영향).
    html = arena_web._render_index()
    # auto 판정이 truthy 체크(t.auto)라 undefined → 기존 경로. '배정'/auto 클래스는 t.auto일 때만.
    assert "if(t.auto){" in html                                 # auto일 때만 분기(없으면 기존 렌더)
    assert "mturns.filter" not in html                           # 필터는 v.valid 대상(구 데이터도 그대로 통과)


def test_preparing_lane_and_caption_in_html():
    # 보드 레인/캡션: 준비 중 전용 라벨(대기와 구분) + 헤더/캡션에 준비 중 명시.
    html = arena_web._render_index()
    assert "게임 준비 중(오라클 로딩)" in html                # 레인 phase 배지
    assert ".lane.preparing" in html and ".phase.preparing" in html
    assert "게임 준비 중 · 오라클 로딩" in html               # 보드 헤더 캡션
    assert "function preparingLaneG" in html                  # 공용 레인 렌더러


def test_preparing_launcher_wait_in_html():
    # 런처: 폴링 상한 45×1s, 준비 중 안내 문구, 시간 초과 시 조용한 포기 대신 정직한 안내.
    html = arena_web._render_index()
    assert "게임 준비 중…" in html                            # 대기 중 안내
    assert "준비가 오래 걸리고 있습니다" in html              # 시간 초과 정직 안내
    assert "function awaitNewRun" in html                     # 새 런만 골라 선택
    assert "function keepAwaitingNewRun" in html              # 초과 후 폴링 유지


# ======================================================================
# 기록 열람: 시드별 그룹(GET /api/seeds) — 측정 조건 단위 coverage
# ======================================================================
def _seed_group(data, seed, game, episodes, max_turns):
    for g in data["groups"]:
        if (g["seed"] == seed and g["game"] == game
                and g["episodes"] == episodes and g["max_turns"] == max_turns):
            return g
    raise AssertionError(f"group not found: {seed}/{game}/{episodes}/{max_turns}")


def test_seeds_grouping_and_union(server):
    # 같은 시드·게임·조건의 두 런(A/B)이 한 그룹으로 묶이고, measured는 완주 슬러그 합집합.
    status, data = _get(server.base, "/api/seeds")
    assert status == 200
    g = _seed_group(data, COV_SEED, "ko-semantle", 1, 10)
    rids = {r["run_id"] for r in g["runs"]}
    assert rids == {SEEDCOV_A, SEEDCOV_B}                     # 두 런이 한 그룹
    # 합집합: opus@low(A) + sonnet@high(A∩B) + codex-luna@low(B). codex-sol는 model_error로 제외.
    assert set(g["measured"]) == {"claude-opus-4-8@low", "claude-sonnet-5@high", "codex-5.6-luna@low"}


def test_seeds_model_error_excluded(server):
    # model_error 에피소드가 있는 참가자(codex-sol)는 '측정 완료'에서 제외(엔진 기준과 동일).
    _, data = _get(server.base, "/api/seeds")
    g = _seed_group(data, COV_SEED, "ko-semantle", 1, 10)
    assert "codex-5.6-sol@low" not in g["measured"]


def test_seeds_condition_split(server):
    # 같은 시드·게임이라도 조건(턴수)이 다르면 별도 그룹(행 분리) — 정직성.
    _, data = _get(server.base, "/api/seeds")
    g10 = _seed_group(data, COV_SEED, "ko-semantle", 1, 10)
    g20 = _seed_group(data, COV_SEED, "ko-semantle", 1, 20)
    assert g10 is not g20
    assert {r["run_id"] for r in g20["runs"]} == {SEEDCOV_C}
    assert set(g20["measured"]) == {"gemini-3-pro@high"}


def test_seeds_latest_run_prefers_done(server):
    # latest_run = 그룹 최신(완주 우선). A/B 중 나중에 시작한 B가 최신.
    _, data = _get(server.base, "/api/seeds")
    g = _seed_group(data, COV_SEED, "ko-semantle", 1, 10)
    assert g["latest_run"] == SEEDCOV_B


def test_seeds_seedless_group(server):
    # seeds 키 없는 구형 런은 seed=None 그룹으로 정직 표기.
    _, data = _get(server.base, "/api/seeds")
    g = _seed_group(data, None, "ko-semantle", 1, 8)
    assert {r["run_id"] for r in g["runs"]} == {SEEDCOV_L}


def test_seeds_rulelab_group_for_prefill(server):
    # semantle 아닌 게임(ko-rulelab) 그룹: 프리필/조건잠금 검증 대상. 카탈로그 대비 미측정 다수 →
    # [미측정 채우기] 버튼이 뜬다(measured < catalog). 그룹 game 필드가 정확히 ko-rulelab이어야
    # 프리필이 그 게임으로 동기화되고 발사 body.game도 rulelab이 된다.
    _, data = _get(server.base, "/api/seeds")
    g = _seed_group(data, RULE_SEED, "ko-rulelab", 1, 15)
    assert {r["run_id"] for r in g["runs"]} == {SEEDCOV_RULE}
    assert set(g["measured"]) == {"claude-opus-4-8@low", "gemini-3-pro@high"}
    # 카탈로그(전 모델)보다 측정이 적어야 미측정이 존재(채우기 버튼 노출 조건).
    catalog = arena_web._client_config()["models"]
    assert len(g["measured"]) < len(catalog)


def test_seeds_preparing_listed_not_measured(server):
    # 준비 중 런(models/ 없음)은 그룹의 runs 리스트엔 뜨되 coverage(measured)엔 미기여.
    _, data = _get(server.base, "/api/seeds")
    # preparing 런은 시드 314159·ko-semantle·2판×15턴 그룹
    g = _seed_group(data, 314159, "ko-semantle", 2, 15)
    rids = {r["run_id"] for r in g["runs"]}
    assert "arena-fixture-preparing" in rids
    assert g["measured"] == []                               # 완주 슬러그 없음


def test_seeds_participant_complete_unit():
    # 웹 자체 완주 판정: summary 유무·status 키·failed·episodes 길이·model_error.
    ok = {"episodes": [{"episode": 1, "solved": True}]}
    assert arena_web._participant_complete(ok, "m@low", set(), 1) is True
    assert arena_web._participant_complete(None, "m@low", set(), 1) is False          # summary 없음
    assert arena_web._participant_complete({"status": "x", "episodes": [{}]}, "m@low", set(), 1) is False  # status 키
    assert arena_web._participant_complete(ok, "m@low", {"m@low"}, 1) is False        # failed 목록
    assert arena_web._participant_complete(ok, "m@low", set(), 2) is False            # episodes 길이 부족
    me = {"episodes": [{"stop_reason": "model_error"}]}
    assert arena_web._participant_complete(me, "m@low", set(), 1) is False            # model_error


# --- 시드별 뷰 클라이언트 표면(HTML) ---
def test_hist_seed_toggle_in_html():
    # 기록 오버레이에 [시드별|날짜별|모델별] 탭 + 시드별 렌더러 + 선택 기억(localStorage).
    html = arena_web._render_index()
    assert 'id="tabSeeds"' in html and '시드별' in html and '날짜별' in html
    assert 'id="histSeeds"' in html
    assert 'function renderHistSeeds' in html
    assert '/api/seeds' in html
    assert "'arena-hist-tab'" in html                        # 선택 기억 키
    assert '정규 세트' in html                                # suiteSeed 그룹 태그
    assert '시드 기록 없음' in html                           # seeds 없는 그룹 표기


def test_hist_seed_prefill_wiring_in_html():
    # coverage 버튼 + 프리필 배선: 미측정 채우기(재활용 안내) · 이 시드로 시작 · 최신 런 열기.
    html = arena_web._render_index()
    assert '미측정 채우기' in html and '이 시드로 시작' in html and '최신 런 열기' in html
    assert 'function openNewPrefill' in html
    assert 'function fillUnmeasured' in html and 'function startWithSeed' in html
    assert '비용 없이 함께 표시' in html                       # 재사용 안내(자동 발사 금지)
    assert '측정 <b>' in html                                 # coverage 라벨(측정 N/카탈로그)


def test_prefill_game_sync_wiring_in_html():
    # 프리필 시 그룹 게임으로 완전 동기화: selGame 설정 + buildGameSeg(세그 활성 버튼·설명·
    # 하단 '게임: …' 문구 재렌더). 발사 body.game=selGame이므로 세그 동기화가 곧 body 동기화.
    html = arena_web._render_index()
    # openNewPrefill 본문에 selGame 설정과 buildGameSeg 호출이 함께 있어야 한다.
    body = html.split('function openNewPrefill', 1)[1].split('async function renderHistRuns', 1)[0]
    assert 'selGame=opt.game' in body
    assert 'buildGameSeg()' in body
    assert 'buildModelGrid()' in body                          # 참가자 목록 렌더


def test_fill_lock_mode_wiring_in_html():
    # 채우기 모드 조건 잠금: 배너 + 잠금 해제 버튼 + 잠금 대상 필드 + 게임 세그 잠금 게이트.
    html = arena_web._render_index()
    # 배너 DOM + 잠금 해제 버튼(문구·핸들러)
    assert 'id="fillBanner"' in html and 'id="fillBannerText"' in html
    assert 'id="fillUnlockBtn"' in html and '조건 잠금 해제' in html
    assert "$('#fillUnlockBtn').onclick=exitFillMode" in html
    # 배너 문구: 조건이 같아야 재활용
    assert '조건이 같아야 기존 완주 결과가 재활용됩니다' in html
    # 잠금 대상: 시드·🎲·반복 수·최대 턴·전부 다시 측정(참가자 선택은 미포함 → 자유)
    assert "FILL_LOCK_IDS=['fSeed','seedDice','fEpisodes','fTurns','fRemeasure']" in html
    assert 'e.disabled=fillMode' in html                       # disabled 배선
    # 게임 세그먼트 잠금: 채우기 모드에서 게임 버튼 클릭 무시 + .locked 클래스
    assert 'if(fillMode) return' in html
    assert "seg.classList.toggle('locked',fillMode)" in html
    # fillUnmeasured는 채우기 모드로 연다
    assert 'fill:true' in html


def test_fill_lock_reset_wiring_in_html():
    # 채우기 모드 누출 방지: 일반 열기·닫기·해제 모두 exitFillMode로 초기화.
    html = arena_web._render_index()
    assert 'function exitFillMode(){ fillMode=false; applyFillLock(); }' in html
    assert 'function openNew(){ exitFillMode();' in html        # 일반 '+ 새 플레이' → 잠금 없음
    assert "if(id==='newOverlay') exitFillMode();" in html      # close()에서 초기화


def test_bulk_select_wiring_in_html():
    # 참가자 일괄 선택: 헤더 [모두 해제] 버튼 + 벤더 그룹 [전체] 토글.
    # [전 모델 선택]은 effort 토글과 역할이 겹쳐 제거(effort 칩 하나 켜기 = 전 모델 선택).
    html = arena_web._render_index()
    assert 'selAllBtn' not in html                              # 버튼 제거 확인(문구는 그룹 토글 주석에 남음)
    assert 'id="selNoneBtn"' in html and '모두 해제' in html
    assert "$('#selNoneBtn').onclick=selectNoneModels" in html
    assert 'function selectNoneModels(){ picked=new Map(); buildModelGrid(); syncCount(); }' in html
    assert 'function toggleFamily(fam){' in html
    # 그룹 토글 버튼이 fam-head에 배선
    assert 'fam-toggle' in html and 'ft.onclick=()=>toggleFamily(fam)' in html
    assert 'function familyAllOn(fam){' in html                 # 켜짐 판정: 전 모델×전 effort
    assert 'if(familyAllOn(fam)){ fam.models.forEach(m=>picked.delete(m.id)); }' in html  # 켜짐 → 해제


def test_group_toggle_full_effort_in_html():
    # 그룹 [전체] 의미론(D): 켜기 = 그룹 전 모델 × 각자 지원 전 effort. 부분 선택이면 꺼짐 판정. 상한 없음.
    html = arena_web._render_index()
    body = html.split('function toggleFamily(fam){', 1)[1].split('function buildBulkEff', 1)[0]
    assert 'const efs=(m.efforts||CFG.efforts)' in body         # 각 모델의 지원 effort
    assert 'efs.forEach(e=>s.add(e))' in body                   # 전 effort 켜기(기본 하나만 담던 옛 정책 아님)
    assert 'defaultEffortOf' not in body                        # 더 이상 기본 effort 하나만 담지 않음
    allon = html.split('function familyAllOn(fam){', 1)[1].split('function toggleFamily', 1)[0]
    assert 'efs.every(e=>s.has(e))' in allon                    # 켜짐 판정도 전 effort 기준
    assert 'const allOn=familyAllOn(fam);' in html              # fam-head 토글이 familyAllOn 사용
    # 그룹 토글은 잠금 대상 아님(fillMode에서도 동작).
    lock_line = [l for l in html.splitlines() if 'FILL_LOCK_IDS=' in l][0]
    assert 'fam-toggle' not in lock_line


def test_bulk_effort_wiring_in_html():
    # 전 모델 effort 토글 줄: 라벨 + 피드백 메시지 + CFG.efforts 기반 상태 칩(.on = 지원 전 모델 켜짐).
    html = arena_web._render_index()
    assert 'id="bulkEff"' in html and '전 모델 effort' in html      # 라벨 있는 컨트롤
    assert 'id="bulkEffMsg"' in html                            # 적용 결과 피드백 슬롯
    assert 'function buildBulkEff(){' in html
    assert "(CFG.efforts||['low','medium','high','xhigh','max']).forEach(e=>{" in html
    assert "el('button','echip'+(bulkEffState(e)?' on':''),e)" in html  # 상태 있는 토글 칩
    assert 'cb.onclick=()=>toggleBulkEffort(e)' in html
    assert 'buildBulkEff();' in html                            # 부트에서 생성
    # 그리드 재렌더마다 토글 상태 동기화(개별 칩 조작 반영) + 오버레이 열 때 재바인딩.
    assert html.count('buildBulkEff();') >= 4                   # 부트 + openNew + openNewPrefill + buildModelGrid


def test_model_label_toggle_wiring_in_html():
    # 모델 라벨 클릭 = 그 모델 전 effort 토글(같은 모델 effort별 비교 워크플로).
    html = arena_web._render_index()
    assert 'function toggleAllEfforts(mid){' in html
    assert 'al.onclick=()=>toggleAllEfforts(m.id);' in html      # 버전 라벨에 배선
    body = html.split('function toggleAllEfforts(mid){', 1)[1].split('function selectNoneModels', 1)[0]
    assert 'if(set && set.size===efs.length){ picked.delete(mid); }' in body  # 전부 켜짐 → 해제
    assert 'efs.forEach(e=>set.add(e))' in body                 # 전 effort 켜기(상한 게이트 없음)
    assert 'partCount()>=CFG.maxParticipants' not in body       # 선택 상한 제거
    assert "(m&&m.efforts)||CFG.efforts" in body                 # 지원 effort는 CFG.models 기준


def test_bulk_effort_policy_in_html():
    # 정책: 토글 — 켜기 = 지원 모델 전부에 e 추가(기존 effort 보존), 끄기 = 전 모델에서 e 제거(빈 선택 해제).
    #        여러 effort 동시 켜기 가능. 선택 상한 없음.
    html = arena_web._render_index()
    # 지원 판정이 CFG.models의 efforts를 읽는다(하드코딩 금지).
    assert 'function supportsEffort(id,e){' in html
    assert '((m&&m.efforts)||CFG.efforts).includes(e)' in html
    # 상태 판정: 지원하는 전 모델이 e를 가졌을 때만 켜짐(부분 적용은 꺼짐 표시 — 정직).
    assert 'function bulkEffState(e){' in html
    body = html.split('function toggleBulkEffort(e){', 1)[1].split('function buildModelGrid', 1)[0]
    assert 'set.add(e); added++;' in body                       # 켜기: 추가(교체 아님 — 기존 effort 보존)
    assert 'picked.set(m.id,new Set([e]))' not in body          # 교체 의미론 제거
    assert 'partCount()>=CFG.maxParticipants' not in body       # 상한 게이트 제거(선택 무제한)
    assert 'capped' not in body                                 # 미적용 집계·'상한 32로' 문구 제거
    assert "set.delete(e); removed++; if(set.size===0) picked.delete(id);" in body  # 끄기: 제거·빈 선택 해제
    assert 'bulkEffMsg(' in body                                # 적용 결과를 명시적으로 보고
    # 전 모델 effort는 잠금 대상 아님 → fillMode에서도 동작.
    lock_line = [l for l in html.splitlines() if 'FILL_LOCK_IDS=' in l][0]
    assert 'bulkEff' not in lock_line


def test_no_selection_cap_in_html():
    # 선택 상한 완전 제거(A): 어떤 클릭 경로에도 partCount 상한 게이트·.dis 비활성·분모 표기가 없다.
    html = arena_web._render_index()
    assert 'partCount()>=CFG.maxParticipants' not in html       # 모든 게이트 제거
    assert "classList.add('dis')" not in html                   # effort 칩 비활성 표기 제거
    assert '상한 32로' not in html                              # 미적용 안내 문구 제거
    # 카운터: "N 참가자"(분모 없음)
    assert "'<b>'+n+'</b> 참가자'" in html
    assert "'</b> / '+CFG.maxParticipants" not in html


def test_workers_wiring_in_html():
    # 최대 동시 실행: 라벨 있는 #fWorkers 입력 + 설명 + body 전송 + 부트 기본값. 측정 키와 무관 → 잠금 대상 아님.
    html = arena_web._render_index()
    assert 'id="fWorkers"' in html and '최대 동시 실행' in html   # 라벨 있는 컨트롤
    assert 'min="1" max="128"' in html                          # 서버 상한과 정렬
    assert '차례를 기다립니다' in html                            # 설명 문구(참가자 초과분 대기)
    assert 'repeat_seed:true, workers}' in html                 # POST body에 workers 포함
    assert 'CFG.defaultWorkers' in html                         # 서버 기본값 사용
    assert "if($('#fWorkers') && CFG.defaultWorkers!=null) $('#fWorkers').value=CFG.defaultWorkers;" in html
    # workers는 채우기 모드 잠금 대상 아님(FILL_LOCK_IDS에 없음).
    lock_line = [l for l in html.splitlines() if 'FILL_LOCK_IDS=' in l][0]
    assert 'fWorkers' not in lock_line


# ======================================================================
# 반복 측정(같은 시드 N회) — 에피소드=다른 문제 은퇴, 반복=안정성
# ======================================================================
def test_post_repeat_seed_adds_flag(server):
    # repeat_seed:true → argv에 --repeat-seed, 응답에 repeat_seed true.
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 3, "max_turns": 15, "seed": 555555, "repeat_seed": True}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["repeat_seed"] is True
    argv = server.spawns[0]
    assert "--repeat-seed" in argv
    assert argv[argv.index("--episodes") + 1] == "3"


def test_post_repeat_seed_default_false(server):
    # repeat_seed 미전달 → 기본 False, --repeat-seed 없음(구형 argv 형태 보존).
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 15}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["repeat_seed"] is False
    assert "--repeat-seed" not in server.spawns[0]


def _one_part():
    return [{"model": "claude-haiku-4-5", "effort": "low"}]


def test_post_workers_valid_adds_flag(server):
    # workers=7 → argv에 --workers 7, 응답에 workers 에코.
    body = {"game": "ko-semantle", "participants": _one_part(),
            "episodes": 1, "max_turns": 15, "workers": 7}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["workers"] == 7
    argv = server.spawns[0]
    assert argv[argv.index("--workers") + 1] == "7"


def test_post_workers_omitted_no_flag(server):
    # workers 미전달 → --workers 없음(엔진 기본 = min(참가자, MAX_WORKERS_DEFAULT)), 에코 None.
    body = {"game": "ko-semantle", "participants": _one_part(), "episodes": 1, "max_turns": 15}
    status, data = _post(server.base, "/api/run", body)
    assert status == 202 and data["workers"] is None
    assert "--workers" not in server.spawns[0]


@pytest.mark.parametrize("bad", [0, 129, -1, "7", 3.5, True])
def test_post_workers_invalid_400(server, bad):
    # workers는 1~128 정수만. 0·129·음수·문자열·실수·bool → 400, 스폰 없음.
    body = {"game": "ko-semantle", "participants": _one_part(),
            "episodes": 1, "max_turns": 15, "workers": bad}
    status, data = _post(server.base, "/api/run", body)
    assert status == 400 and "workers" in data["error"]
    assert server.spawns == []


def test_post_workers_bounds_ok(server):
    # 경계값 1·128 → 통과.
    for w in (1, 128):
        server.spawns.clear()
        status, data = _post(server.base, "/api/run",
                             {"game": "ko-semantle", "participants": _one_part(),
                              "episodes": 1, "max_turns": 15, "workers": w})
        assert status == 202 and data["workers"] == w


# ======================================================================
# 벤치 정지(POST /api/stop) + zombie reap
# ======================================================================
def _seed_running_run(root, run_id="arena-test-stop", pid=99999):
    """정지 검증용: status=running 런의 manifest(+pid)와 index 행을 root에 심는다."""
    rundir = root / run_id
    rundir.mkdir(parents=True, exist_ok=True)
    man = {"run_id": run_id, "game": "ko-semantle", "status": "running",
           "seeds": [12321], "episodes": 1, "max_turns": 5, "pid": pid,
           "participants": [{"model": "claude-haiku-4-5", "effort": "low",
                             "slug": "claude-haiku-4-5@low"}],
           "models": ["claude-haiku-4-5@low"], "started_at": "2026-07-15T10:00:00"}
    (rundir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    ipath = root / "index.json"
    idx = json.loads(ipath.read_text(encoding="utf-8")) if ipath.exists() else {"runs": []}
    idx["runs"].append({"run_id": run_id, "game": "ko-semantle", "status": "running",
                        "episodes": 1, "max_turns": 5, "effort": None,
                        "started_at": "2026-07-15T10:00:00", "finished_at": None,
                        "models": ["claude-haiku-4-5@low"]})
    ipath.write_text(json.dumps(idx), encoding="utf-8")
    return run_id


def test_stop_running_run(server, monkeypatch):
    # running 런 정지: 그룹 SIGTERM(실 시그널 금지 — killpg 몽키패치), manifest·index에 stopped 반영.
    calls = []
    monkeypatch.setattr(arena_web.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    # 대상 프로세스는 이미 없다고 보고(SIGTERM 후 즉시 종료 판정 → SIGKILL 없음, 대기 없음).
    def _gone(pid, sig):
        raise ProcessLookupError()
    monkeypatch.setattr(arena_web.os, "kill", _gone)
    rid = _seed_running_run(server.root, pid=99999)
    status, data = _post(server.base, "/api/stop", {"run_id": rid})
    assert status == 200 and data["ok"] is True and data["status"] == "stopped"
    assert calls and calls[0][0] == 99999 and calls[0][1] == arena_web.signal.SIGTERM
    man = json.loads((server.root / rid / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "stopped" and man.get("finished_at")
    idx = json.loads((server.root / "index.json").read_text(encoding="utf-8"))
    row = next(r for r in idx["runs"] if r["run_id"] == rid)
    assert row["status"] == "stopped" and row.get("finished_at")


def test_stop_idempotent(server, monkeypatch):
    # 두 번 정지: 두 번째는 이미 stopped라 시그널 없이 현재 status 반환(멱등).
    monkeypatch.setattr(arena_web.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(arena_web.os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    rid = _seed_running_run(server.root, pid=99999)
    s1, d1 = _post(server.base, "/api/stop", {"run_id": rid})
    assert s1 == 200 and d1["status"] == "stopped"
    calls = []
    monkeypatch.setattr(arena_web.os, "killpg", lambda pid, sig: calls.append(sig))
    s2, d2 = _post(server.base, "/api/stop", {"run_id": rid})
    assert s2 == 200 and d2["status"] == "stopped" and d2.get("already") is True
    assert calls == []                                   # 이미 끝난 런엔 시그널 안 보냄


def test_stop_escalates_to_sigkill(server, monkeypatch):
    # SIGTERM 후에도 생존하면 대기 초과 시 SIGKILL(대기·폴링 값을 작게 해 실 sleep 최소화).
    calls = []
    monkeypatch.setattr(arena_web.os, "killpg", lambda pid, sig: calls.append(sig))
    monkeypatch.setattr(arena_web.os, "kill", lambda pid, sig: None)   # 항상 생존(kill -0 성공)
    monkeypatch.setattr(arena_web, "_STOP_WAIT_S", 0.03)
    monkeypatch.setattr(arena_web, "_STOP_POLL_S", 0.01)
    rid = _seed_running_run(server.root, pid=99999)
    status, data = _post(server.base, "/api/stop", {"run_id": rid})
    assert status == 200 and data["status"] == "stopped"
    assert arena_web.signal.SIGTERM in calls and arena_web.signal.SIGKILL in calls


def test_stop_unknown_404(server):
    # 존재하지 않는 run_id → 404.
    status, data = _post(server.base, "/api/stop", {"run_id": "arena-nope-xyz"})
    assert status == 404


def test_stop_bad_run_id_400(server):
    # 경로 탈출·형식 오류 run_id → 400(스폰/정지 시도 없음).
    for bad in ("../etc", "a/b", ""):
        status, _ = _post(server.base, "/api/stop", {"run_id": bad})
        assert status == 400


def test_stop_terminal_run_idempotent(server, monkeypatch):
    # 이미 done인 런(픽스처 DONE)에 정지 → 시그널 없이 현재 status(done) 반환.
    calls = []
    monkeypatch.setattr(arena_web.os, "killpg", lambda pid, sig: calls.append(sig))
    status, data = _post(server.base, "/api/stop", {"run_id": DONE})
    assert status == 200 and data["status"] == "done" and data.get("already") is True
    assert calls == []


def test_reap_procs_removes_dead():
    # 종료된 러너 Popen만 회수(defunct 방지), 살아있는 것은 유지.
    class Fake:
        def __init__(self, code): self._c = code
        def poll(self): return self._c
    arena_web._PROCS.clear()
    arena_web._PROCS[111] = Fake(None)   # 살아있음
    arena_web._PROCS[222] = Fake(0)      # 종료됨
    try:
        arena_web._reap_procs()
        assert 111 in arena_web._PROCS and 222 not in arena_web._PROCS
    finally:
        arena_web._PROCS.clear()


def test_spawn_run_stores_popen(monkeypatch):
    # spawn_run이 Popen을 _PROCS에 보관(정지·reap의 2차 소스).
    class FakeProc:
        pid = 54321
        def poll(self): return None
    monkeypatch.setattr(arena_web.subprocess, "Popen", lambda *a, **k: FakeProc())
    arena_web._PROCS.clear()
    try:
        pid = arena_web.spawn_run(["x"])
        assert pid == 54321 and arena_web._PROCS.get(54321) is not None
    finally:
        arena_web._PROCS.clear()


def test_stop_ui_markers_in_html():
    # UI 표면: 정지 버튼·정지 함수·정지 API·정지됨 배지·isLive/topbar 게이트 문자열.
    html = arena_web._render_index()
    assert 'async function stopRun(' in html
    assert "'/api/stop'" in html
    assert 'confirm(' in html                              # 정지 전 확인 1회
    assert "el('button','minibtn stop','정지')" in html    # 실행 중 런 정지 버튼
    assert '정지됨' in html                                # stopped 배지 라벨
    assert '.st.stopped' in html and '.status.stopped' in html
    assert ".minibtn.stop" in html                        # 정지 버튼 스타일
    assert "man.status==='stopped'" in html               # isLive/topbar에 stopped 반영


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_post_bad_repeat_seed_rejected(server, bad):
    body = {"game": "ko-semantle",
            "participants": [{"model": "claude-haiku-4-5", "effort": "low"}],
            "episodes": 1, "max_turns": 5, "repeat_seed": bad}
    status, data = _post(server.base, "/api/run", body)
    assert status == 400 and "repeat_seed" in data["error"]
    assert server.spawns == []


def test_seeds_expose_seeds_list(server):
    # 그룹이 seeds 리스트 전체를 실어 클라이언트가 반복/다문제/단판을 판별할 수 있어야 한다.
    _, data = _get(server.base, "/api/seeds")
    g = next(g for g in data["groups"] if g["seed"] == REPEAT_SEED
             and g["max_turns"] == 15 and {r["run_id"] for r in g["runs"]} == {REPEAT})
    assert g["seeds"] == [REPEAT_SEED, REPEAT_SEED, REPEAT_SEED]


def test_seeds_repeat_vs_consecutive_split(server):
    # 같은 seed_base(555555)라도 seeds 리스트가 다르면 별도 그룹 — 반복 vs 연속 시드 분리(정직).
    _, data = _get(server.base, "/api/seeds")
    same_base = [g for g in data["groups"] if g["seed"] == REPEAT_SEED]
    repeat_g = next(g for g in same_base if all(x == REPEAT_SEED for x in g["seeds"]))
    consec_g = next(g for g in same_base if g["seeds"] == [555555, 555556, 555557])
    assert {r["run_id"] for r in repeat_g["runs"]} == {REPEAT}
    assert {r["run_id"] for r in consec_g["runs"]} == {REPEAT_CONSEC}
    assert repeat_g is not consec_g                       # 뭉치지 않는다


def test_seeds_repeat_count_merged(server):
    # 반복 수만 다른 두 런(1판 + 4회 반복, 같은 시드·턴)이 뷰에서 한 행으로 병합(measurement_key 무변경).
    _, data = _get(server.base, "/api/seeds")
    g = _seed_group(data, MERGE_SEED, "ko-semantle", 4, 50)   # 대표 episodes = 최신 런(4회 반복)
    assert {r["run_id"] for r in g["runs"]} == {MERGE_A, MERGE_B}
    assert g["merged"] is True
    assert g["plays"] == 5                                     # 1 + 4
    assert g["episode_breakdown"] == [{"episodes": 1, "runs": 1}, {"episodes": 4, "runs": 1}]
    # 커버리지 = 합집합(어느 반복 수에서든 완주하면 측정)
    assert set(g["measured"]) == {"claude-opus-4-8@low", "claude-sonnet-5@high", "gemini-3-pro@high"}
    # 채우기 프리필 기본 = 최신 런 조건(4회 반복)
    assert g["episodes"] == 4 and g["latest_run"] == MERGE_B


def test_seeds_merged_single_row(server):
    # 병합 후 909090/semantle 그룹은 정확히 하나(1판만·4회만 별도 행으로 남지 않음).
    _, data = _get(server.base, "/api/seeds")
    hits = [g for g in data["groups"] if g["seed"] == MERGE_SEED and g["game"] == "ko-semantle"]
    assert len(hits) == 1


def test_model_endpoint_exposes_seed(server):
    # 모델별 탭이 (게임·시드·턴)으로 묶어 평균하려면 런마다 seed_base가 실려야 한다(추가 필드).
    _, data = _get(server.base, "/api/model/claude-opus-4-8")
    by = {r["run_id"]: r for r in data["runs"]}
    assert by[MERGE_A]["seed"] == MERGE_SEED and by[MERGE_B]["seed"] == MERGE_SEED


def test_seed_merged_label_wiring_in_html():
    # 병합 행 라벨: 반복 구성 정직 표기(플레이 수 + 브레이크다운) via playLabel/episode_breakdown.
    html = arena_web._render_index()
    assert 'function playLabel(ep){' in html
    assert 'g.merged && Array.isArray(g.episode_breakdown)' in html
    assert "playLabel(b.episodes)+'×'+b.runs" in html
    assert "'플레이('" in html                                # "N플레이(...)" 구성 라벨
    # 채우기 배너도 반복 수 표기(1판/N회 반복) — 최신 런 조건 프리필 명시
    assert "const cond=playLabel(eps)+'×'" in html


def test_model_tab_play_weighted_avg_wiring_in_html():
    # 모델별 탭: (게임·시드·턴·effort) 그룹 + 플레이 가중 평균(반복 횟수 가중) + 표본 수 명시.
    html = arena_web._render_index()
    assert "[r.game,r.seed,r.max_turns,p.effort].join('|')" in html   # 조건 그룹 키
    assert "acc.score[0]+=s.mean_score*w" in html                     # 플레이 가중(episodes=w)
    assert "num(r.episodes)?r.episodes:1" in html                     # 가중치 = 반복 횟수
    assert "plays+'플레이 평균'" in html                              # 표본 수 명시(라벨 없는 합산 금지)


def test_repeat_run_manifest_seeds_served(server):
    # 반복 런 manifest.seeds가 전부 같은 값·길이 3으로 서빙(보드 '시도' 라벨의 근거).
    _, data = _get(server.base, f"/api/run/{REPEAT}")
    s = data["manifest"]["seeds"]
    assert len(s) == 3 and all(x == REPEAT_SEED for x in s)
    assert data["manifest"]["episodes"] == 3


def test_launcher_repeat_field_in_html():
    # 런처: '에피소드 수' → '반복 수'(기본 1) + 안내 문구 + body에 repeat_seed 항상 true.
    html = arena_web._render_index()
    assert '반복 수' in html
    assert '에피소드 수' not in html                         # 옛 라벨 제거
    assert 'value="1"' in html                              # 반복 수 기본 1
    assert '같은 문제(시드)를 N회 풀어 안정성' in html       # 안내 문구
    assert 'repeat_seed:true' in html                       # POST body 항상 포함


def test_seeds_info_label_branch_in_html():
    # seeds 판독 단일 소스: 반복='시도'/'회 반복', 구형 다문제='에피소드'/'연속 시드', 단판='1판'.
    html = arena_web._render_index()
    assert 'function seedsInfo' in html and 'function seedCondLabel' in html
    assert "label:'시도'" in html                            # 반복 런 선택기 라벨
    assert "'회 반복×'" in html                              # 그룹 행: N회 반복
    assert "'문제(연속 시드)×'" in html                      # 그룹 행: 구형 다문제
    assert "seedsInfo(man).label" in html                    # 에피소드 선택기 라벨 배선


# ======================================================================
# 레이아웃 시프트 제거: 조건부 슬롯 공간 예약 + 열 폭 고정
# ======================================================================
def test_layout_slots_reserved_in_html():
    # 조건부 슬롯은 빈 상태에도 자리를 예약. 반응형 3단에서 폰트와 함께 스케일되도록
    # 슬롯 높이를 var(--h-*)로 묶어 라인박스=예약높이 → 폴링/티어 무관 높이 불변.
    html = arena_web._render_index()
    assert "min-height:var(--h-gen)}" in html                # genline(스트림 슬롯)
    assert "min-height:var(--h-hist)}" in html               # histstrip(이력 슬롯)
    assert "min-height:var(--h-cost);line-height:var(--h-cost)}" in html   # costline
    assert "min-height:var(--h-stat);line-height:var(--h-stat)}" in html   # stat-upd(배지 슬롯)
    assert "min-height:var(--h-ticker);line-height:var(--h-ticker)}" in html  # ticker
    assert "line-height:var(--h-gen)}" in html               # gentail 라인박스 = 슬롯 높이


def test_layout_no_display_collapse_and_empty_slots_in_html():
    # 스트림 토글은 display 붕괴가 아니라 visibility로(높이 불변). 빈 배지/이력/코스트도 항상 렌더.
    html = arena_web._render_index()
    assert "gl.style.visibility='hidden'" in html            # display:none 대신 visibility
    assert "gl.style.display='none'" not in html             # 옛 붕괴 경로 제거
    assert "su || el('div','stat-upd')" in html              # 배지 없어도 빈 슬롯 렌더
    assert "min-width:82px" in html                          # 상단 상태칩 폭 고정(텍스트 길이 무관)


def test_responsive_typography_scale_in_html():
    # 반응형 3단(좁음/기본/와이드) + dense: 폰트·간격·슬롯높이를 var로 스케일,
    # 와이드에선 dense도 이력 칩을 노출해 빈 중앙~우측을 활용. 실제 화면 폭에 반응.
    html = arena_web._render_index()
    assert 'content="width=device-width' in html             # 고정 1366 뷰포트 폐기 → 실폭 반응
    assert "@media (max-width:1080px)" in html               # 좁은 화면 컴팩트 단
    assert "@media (min-width:1500px)" in html               # 와이드 확대 단
    assert ".dense .histstrip{display:flex}" in html         # 와이드에선 dense도 이력 노출
    assert "--f-alias:" in html and "--f-big:" in html and "--f-word:" in html   # 타이포 스케일 변수
    assert "font-size:var(--f-alias)" in html                # 모델명 스케일 배선
    assert "font-size:var(--f-big)" in html                  # 최고 순위 숫자 스케일
    assert "grid-template-columns:var(--col1) var(--col2) var(--col3) 1fr var(--col5)" in html  # 열 폭 var
