"""Mindmatch 관전 콘솔 — 단일 파일 관전 대시보드 서버.

여러 LLM이 같은 한국어 단어 게임(꼬맨틀)을 동시에 푸는 것을 실시간으로
지켜보는 스포츠 중계형 스코어보드. HTML/CSS/JS는 이 파일 안에 내장한다.

공개 함수:
    serve(host, port, open_browser, root)  — CLI `python3 -m bench arena serve`가 호출.

API (프론트가 쓰는 JSON 인터페이스):
    GET  /api/index
    GET  /api/run/<run_id>
    GET  /api/run/<run_id>/model/<model>/events?after=N
    GET  /api/run/<run_id>/model/<model>/stream
    GET  /api/model/<model>
    POST /api/run   {game, models[], episodes, max_turns, effort}

데이터 소스는 파일시스템(results/arena). 엔진(bench/arena.py)이 쓴 구조를 읽기만 한다.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import config

# --- 상수 / 검증 --------------------------------------------------------
SEG_RE = re.compile(r"^[A-Za-z0-9._@-]+$")    # 경로 세그먼트 화이트리스트(참가자 slug의 @ 허용)
MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")   # 순수 모델 id(@ 불가)
GAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")  # 게임 이름
EFFORTS = ("low", "medium", "high", "xhigh", "max")
KNOWN_GAMES = ("ko-semantle",)
MAX_PARTICIPANTS = 32   # 참가자 = 모델×effort 조합
MAX_EPISODES = 50
MAX_TURNS = 200
MAX_BODY = 64 * 1024


def _root(root) -> Path:
    return Path(root) if root is not None else (config.RESULTS / "arena")


# --- 파일 읽기 헬퍼 -----------------------------------------------------
def _read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_events(path: Path) -> list:
    """events.jsonl 전체를 파싱해 리스트로. 없으면 빈 리스트."""
    out = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out


def _finished_episodes(events: list) -> set:
    """episode_end 이벤트가 존재하는 에피소드 번호 집합(= 정답 공개 허용 대상)."""
    return {
        ev.get("episode")
        for ev in events
        if isinstance(ev, dict) and ev.get("type") == "episode_end"
    }


def _redact_targets(records: list, finished: set) -> list:
    """진행 중 에피소드의 target을 서빙 단계에서 한 번 더 제거.

    끝난 에피소드(episode_end가 있는 에피소드)의 target만 통과시킨다.
    """
    safe = []
    for rec in records:
        if isinstance(rec, dict) and "target" in rec and rec.get("episode") not in finished:
            rec = {k: v for k, v in rec.items() if k != "target"}
        safe.append(rec)
    return safe


def _strip_target(obj):
    """단일 dict(live 등)에서 target을 무조건 제거(방어)."""
    if isinstance(obj, dict) and "target" in obj:
        return {k: v for k, v in obj.items() if k != "target"}
    return obj


# --- 안전한 경로 해석 ---------------------------------------------------
def _safe_path(root: Path, *segments: str):
    """세그먼트를 화이트리스트로 걸러 root 하위 경로를 만든다. 실패 시 None."""
    for seg in segments:
        if not isinstance(seg, str) or not SEG_RE.match(seg):
            return None
    root = root.resolve()
    candidate = root.joinpath(*segments)
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved != root and root not in resolved.parents:
        return None
    return resolved


# --- 서브프로세스 스폰(테스트에서 monkeypatch 가능하도록 모듈 함수) ----
def spawn_run(argv: list) -> int:
    """`arena run` 서브프로세스를 detached로 띄우고 pid 반환."""
    proc = subprocess.Popen(
        argv,
        cwd=str(config.ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def _validate_run_body(body):
    """POST /api/run 바디 검증. (payload, None) 또는 (None, error_message).

    v2 바디: {game, participants:[{model,effort}], episodes, max_turns}
    (레거시 {models[], effort} 도 관용적으로 받아 participants로 변환한다.)
    참가자 = 모델×effort 조합. slug = <model>@<effort>.
    """
    if not isinstance(body, dict):
        return None, "본문이 JSON 객체가 아님"

    game = body.get("game", "ko-semantle")
    if not isinstance(game, str) or not GAME_RE.match(game) or game not in KNOWN_GAMES:
        return None, f"알 수 없는 게임: {game!r}"

    participants = body.get("participants")
    if participants is None:
        # 레거시 관용 변환: models[] + effort → participants
        models = body.get("models")
        eff = body.get("effort", "low")
        if isinstance(models, list):
            participants = [{"model": m, "effort": eff} for m in models]
    if not isinstance(participants, list) or not (1 <= len(participants) <= MAX_PARTICIPANTS):
        return None, f"participants는 1~{MAX_PARTICIPANTS}개 리스트여야 함"

    catalog = set(config.MODEL_ALIASES)
    seen = set()
    parts_out = []
    slugs = []
    for p in participants:
        if not isinstance(p, dict):
            return None, "participant 형식 오류(객체 아님)"
        m = p.get("model")
        e = p.get("effort", "low")
        if not isinstance(m, str) or not MODEL_RE.match(m):
            return None, f"모델 id 형식 오류: {m!r}"
        if m not in catalog:
            return None, f"카탈로그에 없는 모델: {m!r}"
        if e not in EFFORTS:
            return None, f"effort는 {EFFORTS} 중 하나여야 함: {e!r}"
        allowed = config.model_efforts(m)
        if e not in allowed:
            return None, f"{config.alias(m)}는 effort {e!r}를 지원하지 않음(지원: {allowed})"
        slug = f"{m}@{e}"
        if slug in seen:
            return None, f"참가자 중복: {slug}"
        seen.add(slug)
        parts_out.append({"model": m, "effort": e, "slug": slug})
        slugs.append(slug)

    episodes = body.get("episodes")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or not (1 <= episodes <= MAX_EPISODES):
        return None, f"episodes는 1~{MAX_EPISODES} 정수여야 함"

    max_turns = body.get("max_turns")
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or not (1 <= max_turns <= MAX_TURNS):
        return None, f"max_turns는 1~{MAX_TURNS} 정수여야 함"

    # 스폰: --models 에 slug(model@effort)들을 넘긴다(effort는 slug에 내장, --effort 없음).
    argv = [
        sys.executable, "-m", "bench", "arena", "run",
        "--game", game,
        "--models", *slugs,
        "--episodes", str(episodes),
        "--max-turns", str(max_turns),
    ]
    payload = {
        "game": game, "participants": parts_out, "slugs": slugs,
        "episodes": episodes, "max_turns": max_turns, "argv": argv,
    }
    return payload, None


# --- 서버 설정 주입(카탈로그/기본값을 페이지에 서버렌더) -----------------
def _family(model_id: str):
    """카탈로그 id 패턴으로 (패밀리, 버전 라벨)을 유도. 새 모델도 자동 분류.

    claude-<fam>-<ver...>  → (Fam,       '4.0' / '5')
    codex-<rest>           → ('Codex',   '5.5' / '5.6 Luna')
    gemini-<rest>          → ('Gemini',  '3 Pro')
    gpt-oss-<rest>         → ('GPT-OSS', '120B')
    그 외                   → ('기타',    alias)
    """
    mid = model_id
    if mid.startswith("claude-"):
        parts = mid.split("-")
        fam = parts[1].capitalize() if len(parts) > 1 else "Claude"
        ver = ".".join(parts[2:]) if len(parts) > 2 else config.alias(mid)
        return fam, ver
    for prefix, name in (("codex-", "GPT (Codex)"), ("gemini-", "Gemini"), ("gpt-oss-", "GPT-OSS")):
        if mid.startswith(prefix):
            rest = mid[len(prefix):]
            ver = " ".join(
                t.capitalize() if t.isalpha()
                else t.upper() if any(c.isalpha() for c in t)
                else t
                for t in rest.split("-")
            )
            return name, ver
    return "기타", config.alias(mid)


def _families() -> list:
    """MODEL_ALIASES 선언 순서를 보존하며 패밀리로 그룹핑."""
    fams, index = [], {}
    for mid, alias in config.MODEL_ALIASES.items():
        fam, ver = _family(mid)
        if fam not in index:
            index[fam] = {"name": fam, "models": []}
            fams.append(index[fam])
        index[fam]["models"].append({
            "id": mid, "ver": ver, "alias": alias,
            "efforts": list(config.model_efforts(mid)),
        })
    return fams


def _client_config() -> dict:
    return {
        "models": [
            {"id": mid, "alias": alias, "name": config.model_name(mid),
             "efforts": list(config.model_efforts(mid))}
            for mid, alias in config.MODEL_ALIASES.items()
        ],
        "families": _families(),
        "pilot": list(config.GAME_PILOT_MODELS),
        "efforts": list(EFFORTS),
        "games": list(KNOWN_GAMES),
        "maxParticipants": MAX_PARTICIPANTS,
    }


# --- HTTP 핸들러 --------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "MindmatchArena/2.0"

    def log_message(self, *_a):  # 로그 소음 억제
        pass

    # --- 응답 헬퍼 ---
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status=status)

    @property
    def root(self) -> Path:
        return self.server.root  # type: ignore[attr-defined]

    # --- GET ---
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._html(_render_index())
            return
        if path == "/api/index":
            self._api_index()
            return
        # 세그먼트를 URL 디코드(예: %40→@). 디코드 후에도 SEG_RE·containment로 방어.
        # (%2f→'/' 같은 트래버설은 디코드 후 SEG_RE가 걸러낸다.)
        parts = [unquote(p) for p in path.split("/") if p]
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "run":
            self._api_run(parts[2])
            return
        if (len(parts) == 6 and parts[0] == "api" and parts[1] == "run"
                and parts[3] == "model" and parts[5] == "events"):
            qs = parse_qs(parsed.query)
            try:
                after = int(qs.get("after", ["0"])[0])
            except ValueError:
                after = 0
            self._api_events(parts[2], parts[4], after)
            return
        if (len(parts) == 6 and parts[0] == "api" and parts[1] == "run"
                and parts[3] == "model" and parts[5] == "stream"):
            self._api_stream(parts[2], parts[4])
            return
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "model":
            self._api_model(parts[2])
            return
        self._error(404, "not found")

    # --- POST ---
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._error(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._error(400, "본문 길이 오류")
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._error(400, "JSON 파싱 실패")
            return
        payload, err = _validate_run_body(body)
        if err:
            self._error(400, err)
            return
        try:
            pid = spawn_run(payload["argv"])
        except OSError as exc:
            self._error(500, f"스폰 실패: {exc}")
            return
        self._json({
            "ok": True, "pid": pid,
            "game": payload["game"], "participants": payload["participants"],
            "slugs": payload["slugs"],
            "episodes": payload["episodes"], "max_turns": payload["max_turns"],
        }, status=202)

    # --- API 구현 ---
    def _api_index(self):
        data = _read_json(self.root / "index.json") or {"runs": []}
        self._json(data)

    def _api_run(self, run_id):
        run_dir = _safe_path(self.root, run_id)
        if run_dir is None or not run_dir.is_dir():
            self._error(404, "run 없음")
            return
        manifest = _read_json(run_dir / "manifest.json") or {}
        models_out = {}
        models_dir = run_dir / "models"
        if models_dir.is_dir():
            for md in sorted(models_dir.iterdir()):
                if not md.is_dir() or not SEG_RE.match(md.name):
                    continue
                live = _strip_target(_read_json(md / "live.json"))
                summary = _read_json(md / "summary.json")  # summary.episodes는 전부 끝난 에피소드 → target 허용
                events = _read_events(md / "events.jsonl")
                models_out[md.name] = {
                    "live": live,
                    "summary": summary,
                    "events_count": len(events),
                }
        self._json({"run_id": run_id, "manifest": manifest, "models": models_out})

    def _api_events(self, run_id, model, after):
        path = _safe_path(self.root, run_id, "models", model, "events.jsonl")
        if path is None:
            self._error(404, "not found")
            return
        events = _read_events(path)
        finished = _finished_episodes(events)
        after = max(0, after)
        slice_ = events[after:]
        safe = _redact_targets(slice_, finished)
        self._json({
            "run_id": run_id, "model": model,
            "after": after, "count": len(events), "events": safe,
        })

    def _api_stream(self, run_id, model):
        path = _safe_path(self.root, run_id, "models", model, "stream.json")
        if path is None:
            self._error(404, "not found")
            return
        data = _read_json(path)
        if not isinstance(data, dict):
            data = {"text": "", "done": True}   # absent/unreadable → honest empty-done default
        self._json(data)

    def _api_model(self, model):
        # model은 순수 id(@ 불가). slug의 base(<model>@…)로 참여 여부를 판정.
        if not MODEL_RE.match(model):
            self._error(404, "not found")
            return
        index = _read_json(self.root / "index.json") or {"runs": []}
        runs_out = []
        for run in index.get("runs", []):
            rid = run.get("run_id")
            if not rid:
                continue
            slugs = run.get("models", [])
            matched = [s for s in slugs
                       if isinstance(s, str) and (s == model or s.split("@", 1)[0] == model)]
            if not matched:
                continue
            parts = []
            for s in matched:
                eff = s.split("@", 1)[1] if "@" in s else run.get("effort")
                sdir = _safe_path(self.root, rid, "models", s)
                summary = _read_json(sdir / "summary.json") if sdir else None
                parts.append({"slug": s, "effort": eff, "summary": summary})
            runs_out.append({
                "run_id": rid,
                "game": run.get("game"),
                "status": run.get("status"),
                "episodes": run.get("episodes"),
                "max_turns": run.get("max_turns"),
                "effort": run.get("effort"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "participants": parts,
            })
        self._json({"model": model, "runs": runs_out})


class ArenaServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, root: Path):
        super().__init__(addr, _Handler)
        self.root = root


def make_server(host: str, port: int, root=None) -> ArenaServer:
    return ArenaServer((host, port), _root(root))


def serve(host="127.0.0.1", port=8777, open_browser=True, root=None):
    srv = make_server(host, port, root)
    url = f"http://{host}:{port}/"
    print(f"[arena] 관전 콘솔: {url}  (root={srv.root})", file=sys.stderr)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        srv.server_close()


# --- 인덱스 HTML(서버렌더로 설정 주입) ----------------------------------
def _render_index() -> str:
    return _INDEX_HTML.replace(
        "__ARENA_CONFIG__",
        json.dumps(_client_config(), ensure_ascii=False),
    )


_INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1366, initial-scale=1">
<title>Mindmatch 관전 콘솔</title>
<style>
/* ===================== 시각 정체성 =====================
   팔레트: 다크 우선 방송 스코어보드.
   - 정체성(누구)=차가운 categorical 8색 (이름 라벨이 1차 채널).
   - 근접도(얼마나 가까운가)=따뜻한 앰버 단일 히트 램프.
   - 상태: solved=green, invalid=amber/red, 선두=골드 스포트라이트.
   타이포: 시스템 산세(한글 Apple SD Gothic Neo/Pretendard 폴백).
   라이브로 바뀌는 숫자엔 tabular-nums(폭 흔들림=지터 방지).
======================================================== */
:root{
  --plane:#0d0d0d; --surf:#1a1a19; --surf2:#211f1d; --surf3:#282623;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --line:rgba(255,255,255,.10);
  /* 정체성 8색(cool-first, dark step) */
  --m1:#3987e5; --m2:#199e70; --m3:#9085e9; --m4:#d55181;
  --m5:#26a641; --m6:#e66767; --m7:#c98500; --m8:#d95926;
  /* 히트 램프(앰버, 밝을수록 가깝다) */
  --h0:#7a4d00; --h1:#a56a00; --h2:#c98500; --h3:#ecab2e; --h4:#ffc857;
  /* 상태 */
  --good:#0ca30c; --warn:#fab219; --crit:#e5484d; --gold:#ffc857;
  --font:-apple-system,"Apple SD Gothic Neo","Pretendard",system-ui,"Segoe UI",Roboto,sans-serif;
  --lane-min:52px;
}
:root[data-theme="light"]{
  --plane:#f2f1ec; --surf:#fcfcfb; --surf2:#f6f5f0; --surf3:#eeece5;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7873;
  --grid:#e1e0d9; --axis:#c3c2b7; --line:rgba(11,11,11,.12);
  --m1:#2a78d6; --m2:#1baf7a; --m3:#4a3aa7; --m4:#d84a86;
  --m5:#0f8a2e; --m6:#e34948; --m7:#b07c00; --m8:#d1521f;
  --h0:#dd9a08; --h1:#c07f00; --h2:#9c6400; --h3:#734900; --h4:#4d3100;
  --good:#0a7d0a; --warn:#a86a00; --crit:#c8322f; --gold:#a86a00;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{
  font-family:var(--font); background:var(--plane); color:var(--ink);
  font-size:14px; line-height:1.35; -webkit-font-smoothing:antialiased;
}
button{font-family:inherit;color:inherit;cursor:pointer;border:none;background:none}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--axis);border-radius:6px}
::-webkit-scrollbar-track{background:transparent}

/* ---- 앱 셸: 세로 flex, body 스크롤 없음 ---- */
#app{height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ===== 상단 바 ===== */
header{
  height:56px;flex:0 0 56px;display:flex;align-items:center;gap:14px;
  padding:0 18px;background:var(--surf);border-bottom:1px solid var(--line);
}
.brand{display:flex;align-items:baseline;gap:9px;white-space:nowrap}
.brand .mark{font-weight:800;letter-spacing:.02em;font-size:15px}
.brand .mark b{color:var(--gold)}
.brand .game{font-size:12px;color:var(--muted)}
.brand .game .kr{color:var(--ink2);font-weight:600}

.runsel>button{
  display:flex;align-items:center;gap:9px;height:34px;padding:0 12px;
  background:var(--surf2);border:1px solid var(--line);border-radius:9px;
  font-size:12.5px;max-width:340px;
}
.runsel .rid{font-weight:700;font-variant-numeric:tabular-nums}
.runsel .sub{color:var(--muted);font-size:11px}
.runsel .caret{color:var(--muted);font-size:10px;margin-left:2px}

.epchips{display:flex;gap:5px;align-items:center}
.epchips .lbl{font-size:11px;color:var(--muted);margin-right:2px}
.epchip{
  min-width:26px;height:26px;padding:0 7px;border-radius:7px;font-size:12px;
  background:var(--surf2);border:1px solid var(--line);color:var(--ink2);
  font-variant-numeric:tabular-nums;
}
.epchip.on{background:var(--gold);color:#1a1200;border-color:transparent;font-weight:700}

.status{display:flex;align-items:center;gap:7px;padding:0 11px;height:30px;
  border-radius:20px;font-size:12px;font-weight:700;letter-spacing:.03em}
.status.live{background:rgba(255,200,87,.14);color:var(--gold)}
.status.done{background:var(--surf3);color:var(--ink2)}
.status .dot{width:8px;height:8px;border-radius:50%}
.status.live .dot{background:var(--gold);animation:pulse 1.6s ease-in-out infinite}
.status.done .dot{background:var(--good)}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.7)}}

.meta{display:flex;gap:6px}
.chip{font-size:11px;color:var(--ink2);background:var(--surf2);
  border:1px solid var(--line);border-radius:7px;padding:3px 8px;white-space:nowrap}
.chip b{color:var(--ink);font-variant-numeric:tabular-nums}

.spacer{flex:1}
.actions{display:flex;gap:8px;align-items:center}
.btn{height:34px;padding:0 14px;border-radius:9px;font-size:13px;font-weight:600;
  background:var(--surf2);border:1px solid var(--line)}
.btn:hover{background:var(--surf3)}
.btn.primary{background:var(--gold);color:#1a1200;border-color:transparent}
.btn.primary:hover{filter:brightness(1.06)}
.btn.icon{width:34px;padding:0;font-size:15px}

/* ===== 본문: 보드 + 드로어 ===== */
main{flex:1;min-height:0;display:flex;overflow:hidden}
#board-wrap{flex:1;min-width:0;display:flex;flex-direction:column;padding:12px 16px 14px;overflow:hidden}
.board-head{display:flex;align-items:center;gap:10px;margin-bottom:9px;flex:0 0 auto}
.board-head h2{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700}
.board-head .legend{display:flex;gap:14px;margin-left:auto;font-size:11px;color:var(--muted);align-items:center}
.board-head .legend .k{display:flex;align-items:center;gap:5px}
.board-head .legend .swatch{width:26px;height:9px;border-radius:3px;
  background:linear-gradient(90deg,var(--h0),var(--h2),var(--h4))}
.board-head .legend .arrow{color:var(--ink2)}

#board{flex:0 1 auto;min-height:0;display:flex;flex-direction:column;gap:8px;overflow-y:auto;padding-right:2px}
.dense #board{gap:5px}

/* ---- 접근 궤적 패널(하단 비교 차트) ---- */
#trend{flex:1 1 auto;min-height:140px;max-height:460px;display:flex;flex-direction:column;margin-top:10px;
  background:var(--surf);border:1px solid var(--line);border-radius:12px;padding:9px 14px 7px;overflow:hidden}
.trend-head{display:flex;align-items:center;gap:12px;margin-bottom:4px;flex:0 0 auto;flex-wrap:wrap}
.trend-head .tt{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700}
.trend-head .ax{font-size:11px;color:var(--muted)}
.trend-head .tlegend{display:flex;gap:12px;margin-left:auto;font-size:11px;align-items:center;flex-wrap:wrap}
.trend-head .tlegend .k{display:flex;align-items:center;gap:5px}
.trend-head .tlegend .swatch{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
.trend-head .tlegend .nm{color:var(--ink2)}
.trend-body{flex:1;min-height:0;position:relative}
.trend-svg{display:block;width:100%;height:100%}
.trend-svg .grid{stroke:var(--grid)}
.trend-svg .axis{stroke:var(--axis)}
.trend-svg .lbl{fill:var(--muted);font-size:10px}
.trend-svg .axtitle{fill:var(--ink2);font-size:10.5px}
.trend-svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.trend-svg .dot{stroke:var(--surf);stroke-width:2}
.trend-svg .empty{fill:var(--muted);font-size:12px}
.dense #trend{display:none}
.dense #board{flex:1 1 auto}

/* ---- 레인 ---- */
.lane{
  position:relative;flex:0 0 auto;min-height:var(--lane-min);
  display:grid;align-items:center;
  grid-template-columns:34px 214px 150px 1fr 92px;
  gap:14px;padding:8px 14px 8px 16px;
  background:var(--surf);border:1px solid var(--line);border-radius:12px;
  overflow:hidden;transition:background .18s,border-color .25s,box-shadow .25s;
}
.dense .lane{grid-template-columns:30px 190px 128px 1fr 78px;gap:10px;padding:5px 12px 5px 14px}
.lane:hover{background:var(--surf2);cursor:pointer}
.lane.sel{border-color:var(--gold)}
/* 왼쪽 액센트 바 = 정체성 색 */
.lane::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--mc)}
.lane.leader{box-shadow:inset 0 0 0 1px rgba(255,200,87,.4),0 0 22px -6px rgba(255,200,87,.5)}
.lane.leader::after{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;
  background:linear-gradient(180deg,var(--gold),transparent);animation:shimmer 2.4s linear infinite}
@keyframes shimmer{0%{opacity:.5}50%{opacity:1}100%{opacity:.5}}
.lane.solved{border-color:color-mix(in srgb,var(--good) 60%,transparent)}
.lane.solved::before{background:var(--good)}

/* col1: 순위 위치 */
.pos{font-size:19px;font-weight:800;color:var(--muted);text-align:center;font-variant-numeric:tabular-nums}
.lane.leader .pos{color:var(--gold)}
.dense .pos{font-size:16px}

/* col2: 정체성 */
.who{display:flex;align-items:center;gap:10px;min-width:0}
.who .keydot{width:12px;height:12px;border-radius:50%;background:var(--mc);flex:0 0 auto;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--mc) 22%,transparent)}
.who .names{min-width:0}
.who .aline{display:flex;align-items:center;gap:6px;min-width:0}
.who .alias{font-size:17px;font-weight:750;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* effort 배지(레인·드로어·기록 공용) */
.eff{font-size:10px;font-weight:700;letter-spacing:.02em;padding:1px 6px;border-radius:5px;
  background:var(--surf3);color:var(--ink2);text-transform:uppercase;flex:0 0 auto;white-space:nowrap}
.dense .who .eff{font-size:9px;padding:0 4px}
.who .mid{font-size:10.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:.01em}
.dense .who .alias{font-size:15px}
.dense .who .mid{display:none}
.who .phase{font-size:10px;font-weight:700;letter-spacing:.04em;padding:1px 6px;border-radius:5px;margin-top:2px;display:inline-block}
.phase.running{color:var(--gold);background:rgba(255,200,87,.13)}
.phase.done{color:var(--muted);background:var(--surf3)}
.phase.solvedp{color:var(--good);background:color-mix(in srgb,var(--good) 16%,transparent)}
.dense .who .phase{display:none}

/* col3: 헤드라인 순위 */
.rankcell{text-align:right}
.rankcell .cap{font-size:10px;font-weight:700;letter-spacing:.04em;color:var(--muted);margin-bottom:1px}
.rankcell .big{font-size:32px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;
  color:var(--hc);transition:color .3s;display:inline-block}
.rankcell .big.flash{animation:rankpop .5s ease}
@keyframes rankpop{0%{transform:scale(1)}30%{transform:scale(1.14)}100%{transform:scale(1)}}
.rankcell .unit{font-size:13px;font-weight:600;color:var(--muted);margin-left:1px}
.rankcell .denom{font-size:12px;font-weight:600;color:var(--muted);margin-left:3px}
.rankcell .big.na{color:var(--muted);font-size:22px}
.dense .rankcell .big{font-size:26px}
.dense .rankcell .denom{font-size:10px}
.rankcell .stat-upd{font-size:10px;font-weight:700;margin-top:3px;letter-spacing:.02em;line-height:1.1}
.rankcell .stat-upd.fresh{color:var(--good)}
.rankcell .stat-upd.stall{color:var(--muted)}
.rankcell .stat-upd.stall.long{color:var(--warn)}
.dense .rankcell .stat-upd{font-size:9px}

/* col4: 생성문 · 티커 · 기록 */
.mid-col{display:flex;flex-direction:column;gap:7px;min-width:0}
.ticker{display:flex;align-items:center;gap:9px;min-width:0;font-size:12.5px}
.ticker .verb{font-size:10px;color:var(--muted);letter-spacing:.05em;flex:0 0 auto}
.ticker .word{font-weight:750;font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:150px}
.ticker .sim{color:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums;flex:0 0 auto}
.ticker .sim b{color:var(--ink2);font-weight:600}
:root[data-theme="light"] .ticker .sim b{color:var(--ink2)}
.ticker .rk{color:var(--ink);font-weight:700;font-variant-numeric:tabular-nums;flex:0 0 auto}
.ticker.flash .word{animation:tflash .7s ease}
@keyframes tflash{0%{color:var(--gold)}100%{color:var(--ink)}}
.ticker .bad{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:6px;flex:0 0 auto}
.ticker .bad.fmt{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.ticker .bad.dup{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.ticker .target{margin-left:auto;font-size:12px;color:var(--good);font-weight:700;flex:0 0 auto}
.ticker .target b{font-size:14px}
.dense .ticker .word{font-size:13px}
.dense .ticker .verb{display:none}

/* col5: 턴 */
.turns{text-align:right}
.turns .tn{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
.turns .tn .sep{color:var(--muted);font-weight:500}
.turns .lab{font-size:10px;color:var(--muted);letter-spacing:.04em}
.turns .inv{font-size:10px;color:var(--warn);margin-top:2px;font-variant-numeric:tabular-nums}
.dense .turns .lab{display:none}

/* ===== 우측 상세 드로어 ===== */
#drawer{flex:0 0 0;width:0;background:var(--surf);border-left:1px solid var(--line);
  overflow:hidden;transition:width .22s ease;display:flex;flex-direction:column}
#drawer.open{flex:0 0 384px;width:384px}
.dw-head{flex:0 0 auto;padding:14px 16px 12px;border-bottom:1px solid var(--line)}
.dw-head .top{display:flex;align-items:center;gap:9px}
.dw-head .keydot{width:12px;height:12px;border-radius:50%;background:var(--mc)}
.dw-head .alias{font-size:18px;font-weight:750}
.dw-head .close{margin-left:auto;font-size:18px;color:var(--muted);width:28px;height:28px;border-radius:7px}
.dw-head .close:hover{background:var(--surf3)}
.dw-head .mid{font-size:11px;color:var(--muted);margin-top:2px}
.dw-stats{display:flex;gap:8px;margin-top:12px}
.stat{flex:1;background:var(--surf2);border:1px solid var(--line);border-radius:9px;padding:8px 10px}
.stat .v{font-size:18px;font-weight:750;font-variant-numeric:tabular-nums}
.stat .k{font-size:10px;color:var(--muted);letter-spacing:.03em;margin-top:1px}
.dw-reveal{margin-top:11px;background:color-mix(in srgb,var(--good) 12%,transparent);
  border:1px solid color-mix(in srgb,var(--good) 40%,transparent);border-radius:9px;
  padding:8px 11px;font-size:12px;color:var(--good);display:none}
.dw-reveal.show{display:block}
.dw-reveal b{font-size:15px}

.dw-stream{flex:1;min-height:0;overflow-y:auto;padding:8px 12px 16px}
.dw-stream .lbl{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  padding:8px 4px 6px;position:sticky;top:0;background:var(--surf)}
.turn-row{display:grid;grid-template-columns:30px 1fr auto;gap:9px 9px;align-items:center;
  padding:7px 8px;border-radius:8px}
.turn-row+.turn-row{border-top:1px solid var(--line)}
.turn-row:hover{background:var(--surf2)}
.turn-row .t{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;text-align:right}
.turn-row .g{min-width:0}
.turn-row .g .w{font-size:14px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.turn-row .g .e{font-size:10.5px;color:var(--warn)}
.turn-row.fmt .g .w{color:var(--crit)}
.turn-row .r{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.turn-row .r .s{color:var(--h3)}
:root[data-theme="light"] .turn-row .r .s{color:var(--h2)}
.turn-row .r .rk{font-size:10.5px;color:var(--muted)}
.turn-row .bar{grid-column:1/-1;height:3px;border-radius:2px;background:var(--surf3);margin-top:5px;overflow:hidden}
.turn-row .bar i{display:block;height:100%;border-radius:2px}

/* ===== 오버레이 패널(런처/기록) ===== */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;
  align-items:flex-start;justify-content:center;z-index:50;padding-top:64px}
.overlay.open{display:flex}
.panel{width:660px;max-width:calc(100vw - 40px);max-height:calc(100vh - 120px);
  background:var(--surf);border:1px solid var(--line);border-radius:16px;
  display:flex;flex-direction:column;overflow:hidden;box-shadow:0 24px 60px -18px rgba(0,0,0,.7)}
.panel h3{font-size:15px;font-weight:750}
.panel .ph{display:flex;align-items:center;padding:16px 18px;border-bottom:1px solid var(--line)}
.panel .ph .close{margin-left:auto;font-size:18px;color:var(--muted);width:28px;height:28px;border-radius:7px}
.panel .ph .close:hover{background:var(--surf3)}
.panel .pb{padding:12px 18px;overflow-y:auto}
.panel .pf{padding:14px 18px;border-top:1px solid var(--line);display:flex;gap:10px;align-items:center}
.panel .pf .note{font-size:11.5px;color:var(--muted)}
.panel .pf .grow{flex:1}

.field{margin-bottom:12px}
.field>.k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:7px;display:block}
.modelgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
/* 칩이 곧 선택 — 모든 카드에 상시 노출(높이 균일, 붕 뜸 없음). 항상 2열.
   패밀리 그룹 헤더가 벤더 맥락 담당 → 카드엔 버전 라벨 하나만. */
.fam-head{grid-column:1/-1;display:flex;align-items:baseline;gap:8px;margin:6px 2px 1px}
.fam-head:first-child{margin-top:0}
.fam-head .fn{font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--ink2)}
.fam-head .fc{font-size:10.5px;color:var(--muted)}
.mopt{display:flex;align-items:center;gap:9px;padding:6px 9px;border-radius:9px;align-self:start;
  background:var(--surf2);border:1px solid var(--line);user-select:none}
.mopt.on{border-color:var(--gold);background:color-mix(in srgb,var(--gold) 12%,transparent)}
.mopt .al{flex:0 0 auto;font-weight:750;font-size:13.5px;min-width:26px}
.mopt.on .al{color:var(--gold)}
.effrow{display:flex;gap:4px;flex-wrap:wrap;margin-left:auto;justify-content:flex-end}
.echip{font-size:10.5px;font-weight:600;padding:3px 7px;border-radius:6px;line-height:1;
  background:var(--surf);border:1px solid var(--line);color:var(--ink2)}
.echip:hover{border-color:var(--axis)}
.echip.on{background:var(--gold);color:#1a1200;border-color:transparent}
.echip.dis{opacity:.35;cursor:not-allowed}
.inline{display:flex;gap:16px}
.inline .field{flex:1}
input[type=number],select{width:100%;height:38px;padding:0 11px;border-radius:9px;font-size:14px;
  background:var(--surf2);border:1px solid var(--line);color:var(--ink);font-family:inherit;
  font-variant-numeric:tabular-nums}
.count{font-size:12px;color:var(--muted)}
.count b{color:var(--ink)}
.count.max b{color:var(--gold)}

/* 기록 탭 */
.tabs{display:flex;gap:6px;margin-bottom:14px}
.tab{padding:7px 14px;border-radius:9px;font-size:13px;font-weight:600;background:var(--surf2);border:1px solid var(--line);color:var(--ink2)}
.tab.on{background:var(--surf3);color:var(--ink);border-color:var(--gold)}
.runrow,.mrunrow{display:flex;align-items:center;gap:12px;padding:11px 12px;border-radius:10px;
  border:1px solid var(--line);margin-bottom:8px;cursor:pointer;background:var(--surf2)}
.runrow:hover,.mrunrow:hover{background:var(--surf3)}
.runrow .rid{font-weight:700;font-variant-numeric:tabular-nums;font-size:13px}
.runrow .ms{font-size:11px;color:var(--muted)}
.runrow .st{margin-left:auto;font-size:11px;font-weight:700;padding:3px 9px;border-radius:6px}
.st.running{color:var(--gold);background:rgba(255,200,87,.14)}
.st.done{color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.mrunrow .metrics{margin-left:auto;display:flex;gap:14px;text-align:right}
.mrunrow .metrics .v{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
.mrunrow .metrics .k{font-size:10px;color:var(--muted)}
.selfield{margin-bottom:14px}

/* 빈 상태 */
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;color:var(--muted)}
.empty .big{font-size:19px;color:var(--ink2);font-weight:650}
.empty .sub{font-size:13px}

/* ---- 생성 중(라이브 공개 출력) 라인 + 이력 스트립(레인) ---- */
.genline{display:flex;align-items:center;gap:7px;min-width:0;font-size:11.5px}
.genline .genlab{flex:0 0 auto;font-size:9px;font-weight:700;letter-spacing:.04em;
  color:var(--gold);background:rgba(255,200,87,.13);border-radius:5px;padding:1px 6px;
  animation:pulse 1.6s ease-in-out infinite}
:root[data-theme="light"] .genline .genlab{background:color-mix(in srgb,var(--gold) 14%,transparent)}
.genline.waiting .genlab{color:var(--muted);background:var(--surf3)}
.genline .gentail{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--ink2);letter-spacing:-.01em}
.genline.waiting .gentail{display:none}
.cursor{flex:0 0 auto;display:inline-block;width:6px;height:1em;background:var(--gold);
  vertical-align:text-bottom;animation:blink 1s step-end infinite}
.genline.waiting .cursor{display:none}
@keyframes blink{0%,50%{opacity:1}50.01%,100%{opacity:0}}

.histstrip{display:flex;align-items:center;gap:5px;min-width:0;overflow:hidden}
.histstrip .histlab{flex:0 0 auto;font-size:9px;color:var(--muted);letter-spacing:.03em}
.hchip{flex:0 0 auto;display:inline-flex;align-items:center;gap:4px;max-width:118px;
  font-size:11px;line-height:1;padding:2px 7px;border-radius:6px}
.hchip .hw{min-width:0;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hchip .hr{flex:0 0 auto;font-size:9.5px;font-weight:600;opacity:.82;font-variant-numeric:tabular-nums}
.hchip.inv{background:var(--surf3);color:var(--muted);font-weight:700;padding:2px 6px}

/* dense: 이력 숨김, 생성줄은 최소 '생성 중…' 인디케이터만(멀티라인 tail 제거) */
.dense .histstrip{display:none}
.dense .genline .gentail,.dense .genline .cursor{display:none}
.dense .genline .genlab{font-size:8.5px;padding:0 5px}
.dense .genline:not(.waiting) .genlab::after{content:'…'}

/* ---- 드로어: 생성 중(공개 출력) 라이브 섹션 ---- */
.dw-gen{flex:0 0 auto;display:none;border-bottom:1px solid var(--line);padding:9px 14px 11px}
.dw-gen>.lbl{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--gold);padding:0 2px 7px}
.dw-gen .genfull-wrap{max-height:172px;overflow-y:auto;background:var(--surf2);
  border:1px solid var(--line);border-radius:9px;padding:9px 11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.55;
  color:var(--ink2);white-space:pre-wrap;word-break:break-word}
.dw-gen .genfull{white-space:pre-wrap}
.dw-gen .dw-gen-wait{color:var(--muted);font-size:12px;padding:6px 2px}

.hidden{display:none!important}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="brand">
      <span class="mark">MIND<b>MATCH</b></span>
      <span class="game"><span class="kr">꼬맨틀</span> · 동시 관전</span>
    </div>
    <div class="runsel" id="runsel">
      <button id="runselBtn"><span class="rid" id="runselRid">런 선택</span>
        <span class="sub" id="runselSub"></span><span class="caret">▾</span></button>
    </div>
    <div class="epchips hidden" id="epchips"></div>
    <div class="status done" id="status"><span class="dot"></span><span id="statusTxt">—</span></div>
    <div class="meta" id="meta"></div>
    <div class="spacer"></div>
    <div class="actions">
      <button class="btn primary" id="newBtn">＋ 새 플레이</button>
      <button class="btn" id="histBtn">기록</button>
      <button class="btn icon" id="themeBtn" title="라이트/다크">☾</button>
    </div>
  </header>

  <main>
    <div id="board-wrap">
      <div class="board-head">
        <h2>실시간 순위 · 정답에 가까운 순</h2>
        <div class="legend">
          <span class="k"><span class="swatch"></span><span class="arrow">순위 색 = 정답 근접</span></span>
          <span class="k">1위 = 정답</span>
        </div>
      </div>
      <div id="board"></div>
      <div id="trend" class="trend">
        <div class="trend-head"><span class="tt">접근 궤적</span><span class="ax">턴별 근접도 · 위=정답(1위) · 참가자 색</span><span class="tlegend" id="trendLegend"></span></div>
        <div class="trend-body" id="trendBody"></div>
      </div>
      <div class="empty hidden" id="emptyState">
        <div class="big">아직 관전할 런이 없어요</div>
        <div class="sub">모델을 골라 첫 동시 플레이를 시작해 보세요.</div>
        <button class="btn primary" id="emptyNewBtn">＋ 새 플레이 시작</button>
      </div>
    </div>
    <aside id="drawer"></aside>
  </main>
</div>

<!-- 새 플레이 오버레이 -->
<div class="overlay" id="newOverlay">
  <div class="panel">
    <div class="ph"><h3>새 플레이 시작</h3><button class="close" data-close="newOverlay">✕</button></div>
    <div class="pb">
      <div class="field">
        <span class="k">참가자 선택 · effort 칩을 눌러 추가 <span class="count" id="mcount"></span></span>
        <div class="modelgrid" id="modelGrid"></div>
      </div>
      <div class="inline">
        <div class="field"><span class="k">에피소드 수</span><input type="number" id="fEpisodes" min="1" max="50" value="2"></div>
        <div class="field"><span class="k">최대 턴</span><input type="number" id="fTurns" min="1" max="200" value="15"></div>
      </div>
    </div>
    <div class="pf">
      <span class="note" id="newNote">게임: 꼬맨틀 (ko-semantle)</span>
      <span class="grow"></span>
      <button class="btn" data-close="newOverlay">취소</button>
      <button class="btn primary" id="startBtn">시작</button>
    </div>
  </div>
</div>

<!-- 기록 오버레이 -->
<div class="overlay" id="histOverlay">
  <div class="panel">
    <div class="ph"><h3>기록 열람</h3><button class="close" data-close="histOverlay">✕</button></div>
    <div class="pb">
      <div class="tabs">
        <button class="tab on" id="tabRuns">런별</button>
        <button class="tab" id="tabModels">모델별</button>
      </div>
      <div id="histRuns"></div>
      <div id="histModels" class="hidden">
        <div class="selfield"><span class="k" style="font-size:11px;color:var(--muted)">모델</span>
          <select id="histModelSel"></select></div>
        <div id="histModelRuns"></div>
      </div>
    </div>
  </div>
</div>

<script>
"use strict";
const CFG = __ARENA_CONFIG__;
const REF_DEFAULT = 178;              // 기준어휘 수(fallback). manifest.oracle.reference_words 우선.
const MC = ['--m1','--m2','--m3','--m4','--m5','--m6','--m7','--m8'];
const POLL_MS = 2000;

const S = {
  index:null, runId:null, run:null, episode:1, refWords:REF_DEFAULT,
  ev:{},              // slug -> {events:[], cursor:int}
  colorOf:{},         // slug -> css var
  parts:{},           // slug -> {slug, model, effort, alias}
  focus:null,         // 드로어에 띄운 참가자 slug
  lastTurnKey:{},     // slug -> "ep:turn" (플래시 감지)
  poll:null, loadingRun:false, epPinned:false,  // 사용자가 에피소드를 수동 고정했나
  stream:{},          // slug -> {text,done,turn,model,effort,episode,...} (라이브 공개 출력)
  streamPoll:null,    // 1초 스트림 폴링(구조 폴링과 분리)
};

const $=(s,r=document)=>r.querySelector(s);
const $$=(s,r=document)=>[...r.querySelectorAll(s)];
const el=(t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e;};
const esc=s=>String(s==null?'':s);
async function jget(u){const r=await fetch(u);if(!r.ok)throw new Error(r.status);return r.json();}

/* ---------- 테마 ---------- */
function initTheme(){
  const t=localStorage.getItem('arena-theme')||'dark';
  document.documentElement.setAttribute('data-theme',t);
  $('#themeBtn').textContent=t==='dark'?'☾':'☀';
}
$('#themeBtn').onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const t=cur==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',t);
  localStorage.setItem('arena-theme',t);
  $('#themeBtn').textContent=t==='dark'?'☾':'☀';
  if(S.run) renderAll();
};

/* ---------- 런 인덱스 ---------- */
async function loadIndex(pickFirst){
  S.index=await jget('/api/index');
  const runs=S.index.runs||[];
  if(runs.length===0){showEmpty(true);return;}
  showEmpty(false);
  if(pickFirst && !S.runId) selectRun(runs[0].run_id);
  else if(S.runId) renderTopbar();
}
function showEmpty(on){
  $('#emptyState').classList.toggle('hidden',!on);
  $('#board').classList.toggle('hidden',on);
  $('.board-head').classList.toggle('hidden',on);
  const tr=$('#trend'); if(tr) tr.classList.toggle('hidden',on);
}

/* ---------- 런 선택 + 폴링 ---------- */
async function selectRun(runId){
  if(S.loadingRun) return;
  S.loadingRun=true;
  stopPoll(); stopStreamPoll();
  S.runId=runId; S.focus=null; closeDrawer(); S.ev={}; S.lastTurnKey={}; S.epPinned=false;
  try{
    S.run=await jget('/api/run/'+encodeURIComponent(runId));
  }catch(e){ S.loadingRun=false; return; }
  buildParts();
  assignColors();
  S.refWords=(S.run.manifest&&S.run.manifest.oracle&&S.run.manifest.oracle.reference_words)||REF_DEFAULT;
  S.episode=currentEpisode();
  // 개별 참가자 이벤트 로드 실패가 보드 전체를 비우지 않도록 각각 catch.
  try{ await Promise.all(Object.keys(S.run.models).map(m=>loadEvents(m,true).catch(()=>{}))); }
  finally{ S.loadingRun=false; }
  renderAll();
  if(isLive()){ startPoll(); startStreamPoll(); }
}
function currentEpisode(){
  let ep=1;
  for(const m in S.run.models){const l=S.run.models[m].live; if(l&&l.episode) ep=Math.max(ep,l.episode);}
  return ep;
}
function assignColors(){
  // 정체성 색은 slug(=참가자 엔티티) 순서에 고정. 순위로 바뀌지 않음.
  // 같은 모델의 다른 effort는 slug가 달라 색도 달라진다.
  const order=(S.run.manifest&&S.run.manifest.models)||Object.keys(S.run.models);
  S.colorOf={};
  order.forEach((m,i)=>{S.colorOf[m]=MC[i%MC.length];});
  Object.keys(S.run.models).forEach((m,i)=>{ if(!S.colorOf[m]) S.colorOf[m]=MC[i%MC.length]; });
}
function buildParts(){
  // 각 slug(디렉토리 키)에 대해 {model, effort, alias} 해석.
  // v2: manifest.participants가 권위. 레거시: slug=모델 id, effort=manifest.effort.
  S.parts={};
  const man=S.run.manifest||{};
  const plist=Array.isArray(man.participants)?man.participants:null;
  Object.keys(S.run.models).forEach(slug=>{
    let model=slug, effort=man.effort||'low';
    if(plist){ const p=plist.find(x=>x&&x.slug===slug); if(p){ model=p.model||model; effort=p.effort||effort; } }
    else if(slug.includes('@')){ const a=slug.split('@'); model=a[0]; effort=a[1]||effort; }
    const live=S.run.models[slug].live;   // v2 live/summary는 effort·순수 model 포함
    if(live){ if(live.model) model=live.model; if(live.effort) effort=live.effort; }
    S.parts[slug]={slug, model, effort, alias:modelAlias(model), name:nameOf(model)};
  });
}
function modelAlias(m){const f=CFG.models.find(x=>x.id===m);return f?f.alias:m;}
function nameOf(m){const f=CFG.models.find(x=>x.id===m);return (f&&f.name)?f.name:(f?f.alias:m);}
function aliasOf(slug){ return (S.parts[slug]&&S.parts[slug].name) || nameOf(String(slug).split('@')[0]); }
function effortOf(slug){ return (S.parts[slug]&&S.parts[slug].effort) || (String(slug).includes('@')?slug.split('@')[1]:''); }
function slugLabel(slug){ // 컨텍스트에 참가자 메타가 없을 때(기록 등): "풀네임 · effort"
  const a=String(slug).split('@'); return a.length>1? nameOf(a[0])+' · '+a[1] : nameOf(a[0]);
}
async function loadEvents(model,fresh){
  const cur=fresh?0:(S.ev[model]?S.ev[model].cursor:0);
  const d=await jget(`/api/run/${encodeURIComponent(S.runId)}/model/${encodeURIComponent(model)}/events?after=${cur}`);
  if(fresh) S.ev[model]={events:[],cursor:0};
  const slot=S.ev[model]||(S.ev[model]={events:[],cursor:0});
  slot.events.push(...d.events);
  slot.cursor=d.count;
}
function isLive(){
  if(!S.run) return false;
  const st=(S.run.manifest&&S.run.manifest.status);
  if(st==='running') return true;
  for(const m in S.run.models){const l=S.run.models[m].live; if(l&&l.phase==='running') return true;}
  return false;
}
function startPoll(){ if(S.poll) return; S.poll=setInterval(poll,POLL_MS); }
function stopPoll(){ if(S.poll){clearInterval(S.poll);S.poll=null;} }
function startStreamPoll(){ if(S.streamPoll) return; S.streamPoll=setInterval(streamTick,1000); streamTick(); }
function stopStreamPoll(){ if(S.streamPoll){clearInterval(S.streamPoll);S.streamPoll=null;} }
async function streamTick(){
  if(!S.runId||!S.run||!isLive()){ stopStreamPoll(); return; }
  const running=Object.keys(S.run.models).filter(m=>{
    const l=S.run.models[m].live; return l&&l.phase==='running';
  });
  await Promise.all(running.map(async m=>{
    try{ S.stream[m]=await jget(`/api/run/${encodeURIComponent(S.runId)}/model/${encodeURIComponent(m)}/stream`); }
    catch(e){ /* 이전 상태 유지 */ }
  }));
  paintStreams();
}
function streamState(m){
  const s=S.stream[m];
  const l=S.run&&S.run.models[m]&&S.run.models[m].live;
  const running=l&&l.phase==='running';
  if(!running||!s||s.done){ return {show:false}; }
  const text=s.text||'';
  if(!text.trim()){ return {show:true,waiting:true}; }
  return {show:true,waiting:false,tail:streamTail(text)};
}
function streamTail(text){
  const lines=String(text).replace(/\r/g,'').split('\n').map(x=>x.trimEnd()).filter(x=>x.length);
  let tail=lines.slice(-2).join('   ');
  const MAX=96;
  if(tail.length>MAX) tail='…'+tail.slice(tail.length-MAX);
  return tail;
}
function applyGen(gl,m){
  const st=streamState(m);
  if(!st.show){ gl.style.display='none'; return; }
  gl.style.display='';
  const lab=gl.querySelector('.genlab');
  const tail=gl.querySelector('.gentail');
  if(st.waiting){ gl.classList.add('waiting'); lab.textContent='생성 중…'; tail.textContent=''; }
  else{ gl.classList.remove('waiting'); lab.textContent='생성 중'; tail.textContent=st.tail; }
}
function paintStreams(){
  $$('.lane[data-slug]').forEach(lane=>{
    const gl=lane.querySelector('.genline'); if(gl) applyGen(gl,lane.dataset.slug);
  });
  if(S.focus){
    const l=S.run&&S.run.models[S.focus]&&S.run.models[S.focus].live;
    if(l&&l.phase==='running') paintDrawerStream(S.focus);
  }
}
async function poll(){
  if(!S.runId) return;
  let run; try{run=await jget('/api/run/'+encodeURIComponent(S.runId));}catch(e){return;}
  S.run=run;
  const jobs=[];
  for(const m in run.models){
    const cnt=run.models[m].events_count||0;
    const have=S.ev[m]?S.ev[m].cursor:0;
    if(cnt>have) jobs.push(loadEvents(m,false).catch(()=>{}));
  }
  if(jobs.length) await Promise.all(jobs);
  if(isLive() && !S.epPinned) S.episode=currentEpisode();  // 수동 고정 시 폴링이 덮지 않음
  renderAll();
  try{ await loadIndex(false); }catch(e){}
  if(!isLive()){ stopPoll(); stopStreamPoll(); }
}

/* ---------- 에피소드 뷰 계산(events가 소스) ---------- */
function viewOf(model,ep){
  const slot=S.ev[model]; const evs=slot?slot.events:[];
  const turns=[]; let end=null;
  for(const e of evs){
    if(e.episode!==ep) continue;
    if(e.type==='episode_end'){end=e;}
    else if(e.type==='turn'){turns.push(e);}
  }
  const valid=turns.filter(t=>t.valid);
  const last=turns.length?turns[turns.length-1]:null;
  let bestRank=null;
  for(const t of valid){ if(t.rank!=null && (bestRank==null||t.rank<bestRank)) bestRank=t.rank; }
  if(end&&end.best_rank!=null) bestRank=(bestRank==null)?end.best_rank:Math.min(bestRank,end.best_rank);
  const live=(S.run.models[model]&&S.run.models[model].live)||null;
  const isCur=(ep===currentEpisode());
  let phase='done';
  if(end) phase = end.solved ? 'solved' : 'done';
  else if(live && live.phase==='running') phase='running';
  const lastValid=[...valid].reverse()[0]||null;
  return {
    episode:ep, turns, valid, last, end, bestRank, phase,
    solved: !!(end&&end.solved),
    target: end?end.target:null,     // 서버가 이미 진행 중 에피소드 target을 뺌
    maxTurns:(live&&live.max_turns)|| (S.run.manifest&&S.run.manifest.max_turns) || null,
    invalidCount: turns.filter(t=>!t.valid).length,
    lastSim: last&&last.valid?last.similarity:(lastValid?lastValid.similarity:(live&&isCur?live.last_similarity:null)),
  };
}
function closeness(rank){ // 0..1, rank1=1. 로그 스케일(상위권 확대) — 리더보드에선
  // 순위 1 근처의 차이가 승부처라 선형이면 상위 팩이 뭉개진다.
  if(rank==null) return 0;
  if(rank<=1) return 1;
  const R=Math.max(2,S.refWords);
  return Math.max(0, Math.min(1, 1-Math.log(rank)/Math.log(R)));
}
function heatRGB(c){ // c:0..1 -> [r,g,b] on the amber heat ramp
  const stops=['--h0','--h1','--h2','--h3','--h4'];
  const cs=getComputedStyle(document.documentElement);
  const hex=s=>cs.getPropertyValue(s).trim();
  const x=c*(stops.length-1); const i=Math.min(stops.length-2,Math.floor(x)); const f=x-i;
  const a=hx(hex(stops[i])),b=hx(hex(stops[i+1]));
  return [0,1,2].map(k=>Math.round(a[k]+(b[k]-a[k])*f));
}
function heatColor(c){ const p=heatRGB(c); return `rgb(${p[0]},${p[1]},${p[2]})`; }
function inkOn(rgb){ // 히트색 배경 위 가독 잉크(테마 무관 — 실제 명도로 판정)
  const lum=0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2];
  return lum>150?'#1a1200':'#ffffff';
}
function lerpHex(a,b,t){
  const pa=hx(a),pb=hx(b);
  const r=Math.round(pa[0]+(pb[0]-pa[0])*t),g=Math.round(pa[1]+(pb[1]-pa[1])*t),bl=Math.round(pa[2]+(pb[2]-pa[2])*t);
  return `rgb(${r},${g},${bl})`;
}
function hx(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(c=>c+c).join('');return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}

/* ---------- 렌더 ---------- */
function renderAll(){ renderTopbar(); renderBoard(); if(S.focus) renderDrawer(); }

function renderTopbar(){
  if(!S.runId||!S.run){return;}
  const row=((S.index&&S.index.runs)||[]).find(r=>r.run_id===S.runId)||{};
  const man=(S.run&&S.run.manifest)||{};
  const models=Object.keys(S.run.models||{});
  $('#runselRid').textContent=S.runId.replace(/^arena-/,'');
  $('#runselSub').textContent=`${models.length} 참가자`;
  const live=isLive();
  const st=$('#status'); st.className='status '+(live?'live':'done');
  $('#statusTxt').textContent=live?'LIVE':'완료';
  const nEp=man.episodes||row.episodes||1;
  const ec=$('#epchips');
  if(nEp>1){
    ec.classList.remove('hidden'); ec.innerHTML='';
    ec.appendChild(el('span','lbl','에피소드'));
    for(let i=1;i<=nEp;i++){
      const b=el('button','epchip'+(i===S.episode?' on':''),String(i));
      b.onclick=()=>{S.episode=i; S.epPinned=true; S.focus=null; closeDrawer(); renderBoard();};
      ec.appendChild(b);
    }
  } else ec.classList.add('hidden');
  $('#meta').innerHTML='';
  // effort 배지: 참가자 effort가 전부 같을 때만 단일 표기, 다르면 "혼합".
  const efSet=[...new Set(Object.keys(S.parts).map(s=>S.parts[s].effort))];
  if(efSet.length===1) $('#meta').appendChild(chip('effort',efSet[0]));
  else if(efSet.length>1) $('#meta').appendChild(chip('effort','혼합 '+efSet.length));
  const mt=man.max_turns||row.max_turns;
  if(mt)$('#meta').appendChild(chip('최대',mt+'턴'));
  if(man.oracle&&man.oracle.reference_words)$('#meta').appendChild(chip('기준어휘',man.oracle.reference_words));
}
function chip(k,v){const c=el('span','chip');c.innerHTML=esc(k)+' <b>'+esc(v)+'</b>';return c;}

function renderBoard(){
  const board=$('#board'); if(!S.run){return;}
  const models=Object.keys(S.run.models);
  document.getElementById('board-wrap').classList.toggle('dense',models.length>6);
  document.getElementById('app').classList.toggle('dense',models.length>6);
  const rows=models.map(m=>({m,v:viewOf(m,S.episode)}));
  rows.sort((a,b)=>{
    const ra=a.v.bestRank, rb=b.v.bestRank;
    if(ra==null&&rb==null) return (b.v.lastSim||0)-(a.v.lastSim||0);
    if(ra==null) return 1; if(rb==null) return -1;
    if(ra!==rb) return ra-rb;
    return (b.v.lastSim||0)-(a.v.lastSim||0);
  });
  board.innerHTML='';
  rows.forEach((row,idx)=>board.appendChild(laneEl(row.m,row.v,idx)));
  renderTrend(rows);
}

function statusLine(v){
  if(!v.valid.length) return null;                 // no valid turn yet → omit
  let best=Infinity, impIdx=-1;
  v.valid.forEach((t,i)=>{ if(t.rank!=null && t.rank<best){ best=t.rank; impIdx=i; } });
  const stall = v.valid.length-1 - impIdx;          // valid turns since last best improvement
  const s=el('div','stat-upd');
  if(stall<=0){ s.classList.add('fresh'); s.textContent='▲ 방금 기록 갱신'; }
  else { s.classList.add('stall'); if(stall>=4) s.classList.add('long'); s.textContent='— '+stall+'턴째 제자리'; }
  return s;
}
function laneEl(model,v,idx){
  const lane=el('div','lane');
  lane.dataset.slug=model;
  lane.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  // 큰 숫자 색은 가독성 위해 하한을 둔다(후미 모델도 읽히게).
  lane.style.setProperty('--hc',v.bestRank!=null?heatColor(Math.max(0.28,closeness(v.bestRank))):'var(--muted)');
  const leader=(idx===0)&&v.bestRank!=null&&!v.solved&&isLive();
  if(leader) lane.classList.add('leader');
  if(v.solved) lane.classList.add('solved');
  if(S.focus===model) lane.classList.add('sel');
  lane.onclick=()=>openDrawer(model);

  lane.appendChild(el('div','pos',String(idx+1)));

  const who=el('div','who');
  who.appendChild(el('span','keydot'));
  const names=el('div','names');
  const aline=el('div','aline');
  aline.appendChild(el('span','alias',aliasOf(model)));
  const eff=effortOf(model); if(eff) aline.appendChild(el('span','eff',eff));
  names.appendChild(aline);
  names.appendChild(el('div','mid',model));   // slug 전체(model@effort)
  const ph=el('span','phase '+(v.solved?'solvedp':v.phase),
    v.solved?'정답':(v.phase==='running'?'진행 중':'완료'));
  names.appendChild(ph);
  who.appendChild(names);
  lane.appendChild(who);

  const rc=el('div','rankcell');
  rc.appendChild(el('div','cap','최고 순위'));
  const big=el('div','big'+(v.bestRank==null?' na':''));
  if(v.bestRank!=null){ big.appendChild(document.createTextNode(String(v.bestRank)));
    big.appendChild(el('span','unit','위'));
    big.appendChild(el('span','denom','/ '+S.refWords));
  } else big.textContent='—';
  rc.appendChild(big);
  const su=statusLine(v); if(su) rc.appendChild(su);
  lane.appendChild(rc);

  const mid=el('div','mid-col');
  if(v.phase==='running') mid.appendChild(genLine(model));
  mid.appendChild(tickerEl(model,v));
  const hs=histStrip(model,v); if(hs) mid.appendChild(hs);
  lane.appendChild(mid);

  const turns=el('div','turns');
  const tn=el('div','tn'); const nt=v.last?v.last.turn:(v.end?v.end.turns:0);
  tn.innerHTML='<span>'+esc(nt)+'</span><span class="sep">/'+esc(v.maxTurns||'?')+'</span>';
  turns.appendChild(tn);
  turns.appendChild(el('div','lab','턴'));
  if(v.invalidCount>0) turns.appendChild(el('div','inv','무효 '+v.invalidCount));
  lane.appendChild(turns);

  const tkey=v.last?(v.episode+':'+v.last.turn):'';
  if(S.lastTurnKey[model]&&S.lastTurnKey[model]!==tkey) big.classList.add('flash');
  S.lastTurnKey[model]=tkey;
  return lane;
}

function genLine(model){
  const gl=el('div','genline');
  gl.appendChild(el('span','genlab','생성 중'));
  gl.appendChild(el('span','gentail'));
  gl.appendChild(el('span','cursor'));
  applyGen(gl,model);          // 구조 렌더 직후 현재 스트림을 즉시 반영(1초 공백 방지)
  return gl;
}
function histStrip(model,v){
  if(!v.turns.length) return null;
  const strip=el('div','histstrip');
  strip.appendChild(el('span','histlab','이력'));
  let shownValid=0;
  for(const t of [...v.turns].reverse()){
    if(shownValid>=8) break;
    if(t.valid){
      const c=closeness(t.rank);
      const chip=el('span','hchip');
      const rgb=heatRGB(c); chip.style.background=`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`; chip.style.color=inkOn(rgb);
      chip.appendChild(el('span','hw',t.guess));
      if(t.rank!=null) chip.appendChild(el('span','hr',t.rank+'위'));
      strip.appendChild(chip);
      shownValid++;
    }else{
      const chip=el('span','hchip inv',t.guess?'✕':'무효');
      chip.title=t.guess?'중복 추측':'형식 오류';
      strip.appendChild(chip);
    }
  }
  return strip;
}
function tickerEl(model,v){
  const t=el('div','ticker');
  if(v.solved&&v.target){
    t.appendChild(el('span','verb','정답'));
    const tg=el('span','target'); tg.innerHTML='정답 <b>'+esc(v.target)+'</b>'; t.appendChild(tg);
    return t;
  }
  const last=v.last;
  if(!last){ const w=el('span','word','추측 대기'); w.style.color='var(--muted)'; t.appendChild(w); return t; }
  if(!last.valid){
    const dup=!!last.guess;
    t.appendChild(el('span','verb','최근'));
    if(dup){ t.appendChild(el('span','word',last.guess)); t.appendChild(el('span','bad dup','중복')); }
    else { const w=el('span','word', ((last.raw||'').replace(/\n/g,' ').trim().slice(0,18))||'—'); w.style.color='var(--muted)'; t.appendChild(w); t.appendChild(el('span','bad fmt','형식 오류')); }
    return t;
  }
  t.appendChild(el('span','verb','최근'));
  t.appendChild(el('span','word',last.guess));
  if(last.rank!=null) t.appendChild(el('span','rk','· '+last.rank+'위'));
  const sim=el('span','sim'); sim.innerHTML='· 유사도 <b>'+(last.similarity*100).toFixed(1)+'</b>%'; t.appendChild(sim);
  const tkey=v.episode+':'+last.turn;
  if(S.lastTurnKey[model]&&S.lastTurnKey[model]!==tkey) t.classList.add('flash');
  return t;
}

function renderTrend(rows){
  const host=$('#trendBody'); const legend=$('#trendLegend');
  if(legend) legend.innerHTML='';
  if(!host) return;
  host.innerHTML='';
  if(!rows||!rows.length) return;
  const NS='http://www.w3.org/2000/svg';
  const mk=(tag,attrs,cls)=>{const e=document.createElementNS(NS,tag);if(cls)e.setAttribute('class',cls);for(const k in attrs)e.setAttribute(k,attrs[k]);return e;};
  // 참가자별 시리즈(현재 에피소드) — rows[i].v는 viewOf(m,S.episode) 결과.
  // y=closeness(rank)로 보드 헤드라인과 동일 지표(로그 순위). 위=정답(1위).
  const series=rows.map(r=>({
    color:`var(${S.colorOf[r.m]})`, alias:slugLabel(r.m), solved:!!r.v.solved,
    pts:r.v.valid.filter(t=>t.rank!=null).map(t=>({turn:t.turn, rank:t.rank, c:closeness(t.rank)})),
  }));
  // 범례: 참가자 색 스와치 + 별칭(색=정체성 채널, 글자=텍스트 토큰).
  if(legend) series.forEach(s=>{
    const k=el('span','k'); const sw=el('span','swatch'); sw.style.background=s.color;
    k.appendChild(sw); k.appendChild(el('span','nm',s.alias)); legend.appendChild(k);
  });
  const W=Math.max(220,Math.round(host.clientWidth||640));
  const H=Math.max(120,Math.round(host.clientHeight||160));
  const svg=mk('svg',{viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'none'},'trend-svg');
  const hasAny=series.some(s=>s.pts.length>0);
  if(!hasAny){
    const tx=mk('text',{x:W/2,y:H/2,'text-anchor':'middle','dominant-baseline':'middle'},'empty');
    tx.textContent='추측 대기'; svg.appendChild(tx); host.appendChild(svg); return;
  }
  const mL=44,mR=14,mT=24,mB=30;
  const plotW=Math.max(10,W-mL-mR), plotH=Math.max(10,H-mT-mB);
  let xmax=2; series.forEach(s=>s.pts.forEach(p=>{if(p.turn>xmax)xmax=p.turn;}));
  const X=t=> mL+((t-1)/(xmax-1))*plotW;
  const Y=c=> mT+(1-c)*plotH;   // c=1(정답) → 위
  // 가로 기준선: 위=1위(정답), 아래=바닥
  svg.appendChild(mk('line',{x1:mL,y1:Y(1),x2:mL+plotW,y2:Y(1)},'grid'));
  svg.appendChild(mk('line',{x1:mL,y1:Y(0),x2:mL+plotW,y2:Y(0)},'axis'));
  const l1=mk('text',{x:mL-7,y:Y(1)+3,'text-anchor':'end'},'lbl'); l1.textContent='1위'; svg.appendChild(l1);
  const yt=mk('text',{x:mL,y:12,'text-anchor':'start'},'axtitle'); yt.textContent='가까움 ↑'; svg.appendChild(yt);
  // x축 눈금 + 라벨
  const step=Math.max(1,Math.ceil((xmax-1)/5));
  for(let t=1;t<=xmax;t+=step){
    const tk=mk('text',{x:X(t),y:Y(0)+15,'text-anchor':'middle'},'lbl'); tk.textContent=String(t); svg.appendChild(tk);
  }
  const xt=mk('text',{x:mL+plotW/2,y:H-4,'text-anchor':'middle'},'axtitle'); xt.textContent='턴 →'; svg.appendChild(xt);
  // 참가자별 궤적 폴리라인 + 끝점 도트(solved는 큰 도트)
  series.forEach(s=>{
    if(!s.pts.length) return;
    if(s.pts.length>=2){
      const pl=mk('polyline',{points:s.pts.map(p=>X(p.turn).toFixed(1)+','+Y(p.c).toFixed(1)).join(' ')},'line');
      pl.style.stroke=s.color; svg.appendChild(pl);
    }
    const last=s.pts[s.pts.length-1];
    const dot=mk('circle',{cx:X(last.turn).toFixed(1),cy:Y(last.c).toFixed(1),r:s.solved?5:3.3},'dot');
    dot.style.fill=s.color;
    const tt=mk('title',{}); tt.textContent=`${s.alias} · ${last.turn}턴 · ${last.rank}위`; dot.appendChild(tt);
    svg.appendChild(dot);
  });
  host.appendChild(svg);
}

/* ---------- 드로어(모델 상세) ---------- */
function openDrawer(model){ S.focus=model; $('#drawer').classList.add('open'); renderDrawer(); renderBoard(); }
function closeDrawer(){ S.focus=null; const d=$('#drawer'); if(d)d.classList.remove('open'); if(S.run)renderBoard(); }
function renderDrawer(){
  const model=S.focus; if(!model){return;}
  const v=viewOf(model,S.episode);
  const dw=$('#drawer'); dw.innerHTML='';
  dw.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  const head=el('div','dw-head');
  const top=el('div','top');
  top.appendChild(el('span','keydot'));
  top.appendChild(el('span','alias',aliasOf(model)));
  const deff=effortOf(model); if(deff) top.appendChild(el('span','eff',deff));
  const cl=el('button','close','✕'); cl.onclick=closeDrawer; top.appendChild(cl);
  head.appendChild(top);
  head.appendChild(el('div','mid',model+' · 에피소드 '+S.episode));
  const stats=el('div','dw-stats');
  const brTile=statEl(v.bestRank!=null?v.bestRank+'위 / '+S.refWords:'—','최고 순위');
  if(v.bestRank!=null) brTile.querySelector('.v').style.color=heatColor(Math.max(0.28,closeness(v.bestRank)));
  stats.appendChild(brTile);
  stats.appendChild(statEl(v.lastSim!=null?(v.lastSim*100).toFixed(1):'—','최근 유사도'));
  stats.appendChild(statEl(String(v.valid.length),'유효 추측'));
  stats.appendChild(statEl(String(v.invalidCount),'무효'));
  head.appendChild(stats);
  if(v.target){
    const rev=el('div','dw-reveal show');
    rev.innerHTML='이 에피소드 정답 <b>'+esc(v.target)+'</b> · '+(v.solved?'해결':'미해결');
    head.appendChild(rev);
  }
  dw.appendChild(head);
  const gen=el('div','dw-gen');
  gen.appendChild(el('div','lbl','생성 중 (공개 출력)'));
  const gw=el('div','genfull-wrap');
  gw.appendChild(el('span','genfull'));
  gw.appendChild(el('span','cursor'));
  gen.appendChild(gw);
  gen.appendChild(el('div','dw-gen-wait','아직 공개 출력이 없어요 (생성 중…)'));
  dw.appendChild(gen);
  paintDrawerStream(model);
  const stream=el('div','dw-stream');
  stream.appendChild(el('div','lbl','추측 기록 (공개 출력)'));
  const turnsDesc=[...v.turns].reverse();
  if(turnsDesc.length===0) stream.appendChild(el('div','mid','아직 추측이 없습니다.'));
  turnsDesc.forEach(t=>stream.appendChild(turnRow(t)));
  dw.appendChild(stream);
}
function paintDrawerStream(model){
  const gen=$('#drawer .dw-gen'); if(!gen) return;
  const st=streamState(model);
  if(!st.show){ gen.style.display='none'; return; }
  gen.style.display='block';   // .dw-gen 기본 CSS가 display:none이므로 '' 아닌 명시적 표시
  const wrap=gen.querySelector('.genfull-wrap');
  const wait=gen.querySelector('.dw-gen-wait');
  if(st.waiting){ wrap.style.display='none'; wait.style.display='block'; }
  else{
    wait.style.display='none'; wrap.style.display='';
    gen.querySelector('.genfull').textContent=(S.stream[model]&&S.stream[model].text)||'';
    wrap.scrollTop=wrap.scrollHeight;
  }
}
function statEl(v,k){const s=el('div','stat');s.appendChild(el('div','v',v));s.appendChild(el('div','k',k));return s;}
function turnRow(t){
  const row=el('div','turn-row');
  if(!t.valid&&!t.guess) row.classList.add('fmt');
  row.appendChild(el('div','t','#'+t.turn));
  const g=el('div','g');
  if(t.valid){ g.appendChild(el('div','w',t.guess)); }
  else if(t.guess){ g.appendChild(el('div','w',t.guess)); g.appendChild(el('div','e','중복 추측')); }
  else { g.appendChild(el('div','w', ((t.raw||'').replace(/\n/g,' ').trim().slice(0,22))||'(빈 응답)')); g.appendChild(el('div','e','형식 오류: GUESS 한 줄 필요')); }
  row.appendChild(g);
  const r=el('div','r');
  if(t.valid){ r.innerHTML='<span class="s">유사도 '+(t.similarity*100).toFixed(1)+'%</span><div class="rk">'+t.rank+'위</div>'; }
  else r.innerHTML='<div class="rk">—</div>';
  row.appendChild(r);
  if(t.valid){
    const bar=el('div','bar'); const i=el('i'); const c=closeness(t.rank);
    i.style.width=(c*100).toFixed(1)+'%'; i.style.background=heatColor(c); bar.appendChild(i); row.appendChild(bar);
  }
  return row;
}

/* ---------- 런 선택은 기록 패널로 통합 ---------- */
$('#runselBtn').onclick=()=>openHist();

/* ---------- 새 플레이(참가자 = 모델 × effort) ---------- */
// picked: Map<modelId, Set<effort>>. 모델을 켜면 기본 low 하나가 들어간다.
let picked=new Map(); CFG.pilot.forEach(id=>{const mm=CFG.models.find(x=>x.id===id);const efs=(mm&&mm.efforts)||['low'];picked.set(id,new Set([efs.includes('low')?'low':efs[0]]));});
function partCount(){ let n=0; picked.forEach(s=>n+=s.size); return n; }
function openNew(){ buildModelGrid(); syncCount(); $('#newOverlay').classList.add('open'); }
function toggleEffort(mid,e){
  let set=picked.get(mid);
  if(set && set.has(e)){ set.delete(e); if(set.size===0) picked.delete(mid); }
  else{ if(partCount()>=CFG.maxParticipants) return; if(!set){set=new Set();picked.set(mid,set);} set.add(e); }
  buildModelGrid(); syncCount();
}
function buildModelGrid(){
  // 체크박스 없음 — effort 칩 자체가 선택. 칩은 모든 카드에 상시 노출(높이 균일).
  // 패밀리 그룹 헤더가 벤더 맥락 담당 → 카드엔 버전 라벨 하나만.
  const g=$('#modelGrid'); g.innerHTML='';
  (CFG.families||[]).forEach(fam=>{
    const head=el('div','fam-head');
    head.appendChild(el('span','fn',fam.name));
    head.appendChild(el('span','fc',fam.models.length+'개'));
    g.appendChild(head);
    fam.models.forEach(m=>{
      const set=picked.get(m.id);
      const on=!!(set && set.size);
      const opt=el('div','mopt'+(on?' on':'')); opt.title=m.id;  // 전체 id는 title 툴팁만
      opt.appendChild(el('span','al',m.ver));                    // 버전 라벨 하나만
      const er=el('div','effrow');
      (m.efforts||CFG.efforts).forEach(e=>{
        const active=!!(set && set.has(e));
        const cb=el('button','echip'+(active?' on':''),e);
        cb.onclick=()=>toggleEffort(m.id,e);
        if(!active && partCount()>=CFG.maxParticipants) cb.classList.add('dis');
        er.appendChild(cb);
      });
      opt.appendChild(er);
      g.appendChild(opt);
    });
  });
}
function syncCount(){
  const n=partCount();
  const c=$('#mcount'); c.innerHTML='<b>'+n+'</b> / '+CFG.maxParticipants+' 참가자';
  c.classList.toggle('max',n>=CFG.maxParticipants);
  $('#startBtn').disabled=n===0;
  $('#startBtn').style.opacity=n===0?.5:1;
}
function currentParticipants(){
  const out=[];
  CFG.models.forEach(m=>{ const set=picked.get(m.id); if(set) (m.efforts||CFG.efforts).forEach(e=>{ if(set.has(e)) out.push({model:m.id,effort:e}); }); });
  return out;
}
$('#startBtn').onclick=async()=>{
  const participants=currentParticipants();
  if(participants.length===0) return;
  const body={game:'ko-semantle',participants,
    episodes:+$('#fEpisodes').value||1, max_turns:+$('#fTurns').value||1};
  $('#startBtn').disabled=true; $('#newNote').textContent='시작 중…';
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok){ $('#newNote').textContent='오류: '+(d.error||r.status); $('#startBtn').disabled=false; return; }
    close('newOverlay');
    S.runId=null;
    for(let i=0;i<20;i++){ await new Promise(rs=>setTimeout(rs,700)); await loadIndex(true); if(S.runId){break;} }
  }catch(e){ $('#newNote').textContent='요청 실패'; }
  finally{ $('#newNote').textContent='게임: 꼬맨틀 (ko-semantle)'; $('#startBtn').disabled=false; }
};

/* ---------- 기록 ---------- */
async function openHist(){ $('#histOverlay').classList.add('open'); await renderHistRuns(); buildHistModelSel(); }
async function renderHistRuns(){
  const box=$('#histRuns'); box.innerHTML='';
  const runs=(S.index&&S.index.runs)||[];
  runs.forEach(r=>{
    const row=el('div','runrow');
    row.appendChild(el('span','rid',r.run_id.replace(/^arena-/,'')));
    row.appendChild(el('span','ms',(r.models||[]).map(slugLabel).join(', ')+` · ${r.episodes}ep · ${r.max_turns}턴`));
    row.appendChild(el('span','st '+(r.status==='running'?'running':'done'),r.status==='running'?'LIVE':'완료'));
    row.onclick=()=>{ close('histOverlay'); selectRun(r.run_id); };
    box.appendChild(row);
  });
}
function buildHistModelSel(){
  const sel=$('#histModelSel'); sel.innerHTML='';
  const seen=new Set();
  ((S.index&&S.index.runs)||[]).forEach(r=>(r.models||[]).forEach(s=>seen.add(String(s).split('@')[0])));
  [...seen].forEach(m=>{const o=el('option',null,nameOf(m)+'  ('+m+')');o.value=m;sel.appendChild(o);});
  sel.onchange=()=>renderHistModel(sel.value);
  if(seen.size) renderHistModel([...seen][0]);
}
async function renderHistModel(model){
  const box=$('#histModelRuns'); box.innerHTML='<div style="color:var(--muted);font-size:12px">불러오는 중…</div>';
  let d; try{d=await jget('/api/model/'+encodeURIComponent(model));}catch(e){box.innerHTML='';return;}
  box.innerHTML='';
  const rows=[]; (d.runs||[]).forEach(r=>(r.participants||[]).forEach(p=>rows.push({r,p})));
  if(!rows.length){box.innerHTML='<div style="color:var(--muted);font-size:12px">참여한 런이 없습니다.</div>';return;}
  // effort별 비교가 목적 → 참가자(런×effort) 단위 행
  rows.forEach(({r,p})=>{
    const row=el('div','mrunrow');
    const left=el('div');
    const rl=el('div','rid'); rl.appendChild(document.createTextNode(r.run_id.replace(/^arena-/,'')));
    if(p.effort) rl.appendChild(el('span','eff',p.effort));
    left.appendChild(rl);
    left.appendChild(el('div','ms',`${r.episodes}ep · ${r.max_turns}턴`));
    row.appendChild(left);
    const s=p.summary||{};
    const met=el('div','metrics');
    met.appendChild(metric(s.solve_rate!=null?(s.solve_rate*100).toFixed(0)+'%':'—','해결률'));
    met.appendChild(metric(s.mean_score!=null?(s.mean_score*100).toFixed(0):'—','점수'));
    met.appendChild(metric(s.median_turns!=null?s.median_turns:'—','중앙턴'));
    row.appendChild(met);
    row.onclick=()=>{ close('histOverlay'); selectRun(r.run_id); };
    box.appendChild(row);
  });
}
function metric(v,k){const m=el('div');m.appendChild(el('div','v',v));m.appendChild(el('div','k',k));return m;}
$('#tabRuns').onclick=()=>{$('#tabRuns').classList.add('on');$('#tabModels').classList.remove('on');$('#histRuns').classList.remove('hidden');$('#histModels').classList.add('hidden');};
$('#tabModels').onclick=()=>{$('#tabModels').classList.add('on');$('#tabRuns').classList.remove('on');$('#histModels').classList.remove('hidden');$('#histRuns').classList.add('hidden');};

/* ---------- 오버레이 공통 ---------- */
function close(id){ $('#'+id).classList.remove('open'); }
$$('[data-close]').forEach(b=>b.onclick=()=>close(b.dataset.close));
$$('.overlay').forEach(o=>o.addEventListener('click',e=>{if(e.target===o)o.classList.remove('open');}));
$('#newBtn').onclick=openNew;
$('#emptyNewBtn').onclick=openNew;
$('#histBtn').onclick=openHist;
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$$('.overlay.open').forEach(o=>o.classList.remove('open'));if(S.focus)closeDrawer();}});

/* ---------- 리사이즈: 궤적 SVG 재측정 ---------- */
let _trendRAF=null;
window.addEventListener('resize',()=>{
  if(!S.run) return;
  if(_trendRAF) cancelAnimationFrame(_trendRAF);
  _trendRAF=requestAnimationFrame(renderBoard);
});

/* ---------- 부트 ---------- */
initTheme();
loadIndex(true);
</script>
</body>
</html>
"""
