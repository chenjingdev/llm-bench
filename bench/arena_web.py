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
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import arena
from . import config

# --- 상수 / 검증 --------------------------------------------------------
SEG_RE = re.compile(r"^[A-Za-z0-9._@-]+$")    # 경로 세그먼트 화이트리스트(참가자 slug의 @ 허용)
MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")   # 순수 모델 id(@ 불가)
GAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")  # 게임 이름
EFFORTS = ("low", "medium", "high", "xhigh", "max")
KNOWN_GAMES = ("ko-semantle", "ko-rulelab", "ko-maze", "ko-minefield")
# 게임 카탈로그: 한글명 · 한 줄 설명 · 런처 기본 max_turns(게임별 엔진 기본에 정렬).
# semantle는 기존 런처 기본(15)을 보존한다(무변경). 신규 게임은 엔진 DEFAULT_MAX_TURNS.
GAME_META = {
    "ko-semantle":  {"kr": "꼬맨틀",          "desc": "숨은 목표 단어에 의미(임베딩)로 다가가는 추론 게임",       "max_turns": 15},
    "ko-rulelab":   {"kr": "비밀 규칙 연구소", "desc": "실험(TEST)으로 숨은 함수 규칙 f(a,b)를 알아내 5문항 예측", "max_turns": 15},
    "ko-maze":      {"kr": "숨은 지도 탐험",   "desc": "안개 낀 7×7 미로에서 방위 단서만으로 목표 칸 찾기",        "max_turns": 40},
    "ko-minefield": {"kr": "의미 지뢰밭",     "desc": "숨은 지뢰 의미 영역을 피해 목표 단어에 접근(목숨 3)",       "max_turns": 40},
}
# 정답류(비누출) 필드: 진행 중 에피소드에서 서버가 제거. episode_end에만 노출된다.
_SECRET_FIELDS = ("target", "mines")
MAX_EPISODES = 50
MAX_TURNS = 200
MAX_BODY = 64 * 1024
MAX_WORKERS = 128       # 최대 동시 실행 상한(자원 보호는 이 값으로, 선택은 막지 않음)


def _total_combos() -> int:
    """가능한 최대 참가자 수 = 카탈로그 전 모델의 지원 effort 수 합(모델×effort 조합).

    선택을 막는 상한이 아니라 중복 없는 물리적 최대다. config에서 계산한다(하드코딩 금지).
    """
    return sum(len(config.model_efforts(mid)) for mid in config.MODEL_ALIASES)


def _root(root) -> Path:
    return Path(root) if root is not None else (config.RESULTS / "arena")


# --- 파일 읽기 헬퍼 -----------------------------------------------------
def _num(x) -> bool:
    """정렬 키로 쓸 수 있는 실수인가(bool 제외)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


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
    """진행 중 에피소드의 정답류(target·mines)를 서빙 단계에서 한 번 더 제거.

    끝난 에피소드(episode_end가 있는 에피소드)의 정답류만 통과시킨다. minefield의
    지뢰(mines)·maze의 목표 좌표(target)도 target과 동일하게 완료 에피소드만 통과한다.
    """
    safe = []
    for rec in records:
        if (isinstance(rec, dict) and rec.get("episode") not in finished
                and any(f in rec for f in _SECRET_FIELDS)):
            rec = {k: v for k, v in rec.items() if k not in _SECRET_FIELDS}
        safe.append(rec)
    return safe


def _strip_target(obj):
    """단일 dict(live 등)에서 정답류(target·mines)를 무조건 제거(방어)."""
    if isinstance(obj, dict) and any(f in obj for f in _SECRET_FIELDS):
        return {k: v for k, v in obj.items() if k not in _SECRET_FIELDS}
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


# 스폰한 러너 Popen 보관(pid→Popen). 정지 시 poll()로 zombie/생존을 정확히 판정하고,
# 종료된 프로세스를 reap해 defunct(zombie)를 방지한다. 서버 재시작 시 비지만, 그때는
# manifest["pid"]가 1차 소스이므로 정지는 계속 동작한다.
_PROCS: dict = {}
_STOP_WAIT_S = 3.0     # SIGTERM 후 최대 대기(초) — 초과 시 SIGKILL
_STOP_POLL_S = 0.1     # 생존 확인 폴링 간격(초)


# --- 서브프로세스 스폰(테스트에서 monkeypatch 가능하도록 모듈 함수) ----
def spawn_run(argv: list) -> int:
    """`arena run` 서브프로세스를 detached로 띄우고 pid 반환. Popen은 _PROCS에 보관(reap·정지용)."""
    proc = subprocess.Popen(
        argv,
        cwd=str(config.ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _PROCS[proc.pid] = proc
    return proc.pid


def _reap_procs() -> None:
    """종료된 러너 Popen을 회수(defunct 방지). poll()이 None 아니면 OS 자식 테이블에서 수거된다."""
    for pid, proc in list(_PROCS.items()):
        try:
            if proc.poll() is not None:
                _PROCS.pop(pid, None)
        except Exception:
            _PROCS.pop(pid, None)


def _alive(pid: int, proc) -> bool:
    """프로세스 생존 여부. Popen이 있으면 poll() 우선(zombie도 kill -0엔 살아있다고 나오므로)."""
    if proc is not None:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # 존재하나 시그널 권한 없음 → 살아있다고 간주


def _stop_process(pid: int, proc) -> None:
    """프로세스 그룹을 SIGTERM → 최대 _STOP_WAIT_S 대기 → 생존 시 SIGKILL(best-effort·멱등)."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return   # 이미 종료됐거나 권한 없음 — 멱등 처리
    deadline = time.monotonic() + _STOP_WAIT_S
    while time.monotonic() < deadline:
        if not _alive(pid, proc):
            return
        time.sleep(_STOP_POLL_S)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _write_json_atomic(path: Path, obj) -> None:
    """임시 파일에 쓴 뒤 os.replace로 원자적 교체(부분 쓰기로 인한 손상 방지)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _mark_index_stopped(root: Path, run_id: str, finished_at: str) -> None:
    """index.json에서 해당 run의 status='stopped'·finished_at 갱신(있을 때만, 원자적 기록)."""
    ipath = root / "index.json"
    data = _read_json(ipath)
    if not isinstance(data, dict):
        return
    runs = data.get("runs")
    if not isinstance(runs, list):
        return
    changed = False
    for run in runs:
        if isinstance(run, dict) and run.get("run_id") == run_id:
            run["status"] = "stopped"
            run["finished_at"] = finished_at
            changed = True
    if changed:
        _write_json_atomic(ipath, data)


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
    max_parts = _total_combos()
    if not isinstance(participants, list) or not (1 <= len(participants) <= max_parts):
        return None, f"participants는 1~{max_parts}개(카탈로그 총 조합 수) 리스트여야 함"

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

    # 측정 경제: seed(정규 문제 세트 고정 seed, 선택) · reuse(동일 조건 완주 결과 재사용, 기본 True).
    # seed 있으면 정규 세트(재현 가능), 없으면 엔진이 시간 기반 새 문제.
    seed = body.get("seed")
    if seed is not None:
        if (not isinstance(seed, int) or isinstance(seed, bool)
                or seed < 0 or seed > 2**63 - 1):
            return None, "seed는 0 이상의 정수여야 함"
    reuse = body.get("reuse", True)
    if not isinstance(reuse, bool):
        return None, "reuse는 true/false여야 함"

    # 반복 측정: repeat_seed=True면 엔진이 seeds=[seed_base]*episodes(전판 동일 문제)로 돌린다.
    # (에피소드=다른 문제 은퇴 → 같은 시드 N회 반복으로 안정성 측정.) 선택, 기본 False.
    repeat_seed = body.get("repeat_seed", False)
    if not isinstance(repeat_seed, bool):
        return None, "repeat_seed는 true/false여야 함"

    # 최대 동시 실행(선택): 자원 보호용 실행 파라미터. 측정 키(game/seeds/max_turns)와 무관하므로
    # 채우기 모드에서도 조정 가능. 1~MAX_WORKERS. CLI `arena run --workers`로 전달.
    workers = body.get("workers")
    if workers is not None:
        if (not isinstance(workers, int) or isinstance(workers, bool)
                or not (1 <= workers <= MAX_WORKERS)):
            return None, f"workers는 1~{MAX_WORKERS} 정수여야 함"

    # 스폰: --models 에 slug(model@effort)들을 넘긴다(effort는 slug에 내장, --effort 없음).
    argv = [
        sys.executable, "-m", "bench", "arena", "run",
        "--game", game,
        "--models", *slugs,
        "--episodes", str(episodes),
        "--max-turns", str(max_turns),
    ]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if not reuse:                      # 재사용 끔(전부 다시 측정) → --no-reuse
        argv += ["--no-reuse"]
    if repeat_seed:                    # 같은 문제 N회 반복(안정성) → --repeat-seed
        argv += ["--repeat-seed"]
    if workers is not None:            # 최대 동시 실행 → --workers N
        argv += ["--workers", str(workers)]
    payload = {
        "game": game, "participants": parts_out, "slugs": slugs,
        "episodes": episodes, "max_turns": max_turns,
        "seed": seed, "reuse": reuse, "repeat_seed": repeat_seed,
        "workers": workers, "argv": argv,
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
        "gameMeta": {g: dict(GAME_META[g]) for g in KNOWN_GAMES},
        "maxParticipants": _total_combos(),
        "defaultWorkers": arena.MAX_WORKERS_DEFAULT,
        # 정규 문제 세트 고정 seed(측정 경제). 엔진 config에서 안전하게 읽는다(부재 시 기본값).
        "suiteSeed": getattr(config, "ARENA_SUITE_SEED_BASE", 314159),
    }


# --- 기록 열람: 런 요약(참가 수·1등·업적) ------------------------------
def _slug_model_effort(slug: str, pmap: dict, summary, run_effort):
    """slug → (순수 model id, effort). manifest.participants > summary > slug@ > 런 effort."""
    model = effort = None
    p = pmap.get(slug)
    if isinstance(p, dict):
        model, effort = p.get("model"), p.get("effort")
    if isinstance(summary, dict):
        model = summary.get("model") or model
        effort = summary.get("effort") or effort
    if not model:
        model = slug.split("@", 1)[0]
    if not effort:
        effort = slug.split("@", 1)[1] if "@" in slug else run_effort
    return model, effort


def _achievement(game: str, summary: dict) -> str:
    """승자 summary.episodes로 게임별 한국어 한 줄 업적을 조립(해결/미해결 분기)."""
    eps = summary.get("episodes")
    eps = [e for e in eps if isinstance(e, dict)] if isinstance(eps, list) else []
    solved = [e for e in eps if e.get("solved")]

    def _min(key, src):
        vals = [e.get(key) for e in src if e.get(key) is not None]
        return min(vals) if vals else None

    if game == "ko-maze":
        if solved:
            n = _min("turns", solved)
            return f"도착 · {n}턴" if n is not None else "도착"
        d = _min("min_dist", eps)
        return f"최접근 {d}칸" if d is not None else "미도착"
    if game == "ko-rulelab":
        full = [e for e in eps if e.get("correct") == 5]
        if full:
            n = _min("experiments", full)
            return f"5/5 적중 · 실험 {n}회" if n is not None else "5/5 적중"
        answered = [e for e in eps if e.get("correct") is not None]
        if answered:
            best = max(answered, key=lambda e: e.get("correct") or 0)
            return f"{best.get('correct') or 0}/5 적중"
        return "미답변"
    # ko-semantle · ko-minefield: best_rank(순위) 계열
    if solved:
        n = _min("turns", solved)
        return f"정답 · {n}턴" if n is not None else "정답"
    r = _min("best_rank", eps)
    if game == "ko-minefield" and eps and all(e.get("stop_reason") == "mined" for e in eps):
        return f"폭사 · 최고 {r}위" if r is not None else "폭사"
    return f"최고 {r}위" if r is not None else "미해결"


def _run_summary(root: Path, run: dict):
    """index 런 항목 하나에 대한 (participants_count, winner|None).

    winner: summary.json이 있고 실패 마킹(summary.status·failed_participants) 없는 참가자 중
    (mean_score↓, median_turns↑, usage.duration_ms↑, usage.output_tokens↑, slug↑) 1위.
    usage 결측(구형 런) 단계는 무한대로 취급해 그 단계에서 최하로 밀되 그 위 순서는 유지.
    전원 무득점·미해결이면 None.
    """
    run_id = run.get("run_id")
    slugs = run.get("models")
    if not isinstance(slugs, list):
        slugs = []
    count = sum(1 for s in slugs if isinstance(s, str))
    if not isinstance(run_id, str):
        return count, None
    game = run.get("game") or "ko-semantle"
    run_effort = run.get("effort")

    manifest = {}
    mpath = _safe_path(root, run_id, "manifest.json")
    if mpath:
        manifest = _read_json(mpath) or {}
    pmap = {p["slug"]: p for p in (manifest.get("participants") or [])
            if isinstance(p, dict) and isinstance(p.get("slug"), str)}
    failed = {f.get("slug") for f in (manifest.get("failed_participants") or [])
              if isinstance(f, dict)}

    cands = []
    any_score = any_solved = False
    for slug in slugs:
        if not isinstance(slug, str):
            continue
        spath = _safe_path(root, run_id, "models", slug, "summary.json")
        summary = _read_json(spath) if spath else None
        if not isinstance(summary, dict) or "status" in summary or slug in failed:
            continue
        ms = summary.get("mean_score") or 0
        if ms > 0:
            any_score = True
        eps = summary.get("episodes")
        if isinstance(eps, list) and any(isinstance(e, dict) and e.get("solved") for e in eps):
            any_solved = True
        model, effort = _slug_model_effort(slug, pmap, summary, run_effort)
        mt = summary.get("median_turns")
        usage = summary.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        dur = usage.get("duration_ms")
        otk = usage.get("output_tokens")
        cands.append((-ms,
                      mt if _num(mt) else float("inf"),
                      dur if _num(dur) else float("inf"),   # usage 결측 → 최하
                      otk if _num(otk) else float("inf"),
                      slug, model, effort, summary))
    if not cands or (not any_score and not any_solved):
        return count, None
    cands.sort(key=lambda c: (c[0], c[1], c[2], c[3], c[4]))
    _, _, _, _, _, model, effort, summary = cands[0]
    return count, {"model": model, "effort": effort,
                   "achievement": _achievement(game, summary)}


# --- 기록 열람: 시드별 그룹(측정 조건 단위 coverage) --------------------
def _participant_complete(summary, slug, failed_slugs, episodes) -> bool:
    """참가자 '측정 완료' 판정 — 엔진 _participant_complete와 같은 기준(웹 자체 구현).

    완료 = summary.json 존재 && summary에 'status' 키 없음 && failed_participants에 없음
    && summary.episodes 길이 == 런 episodes && 어느 에피소드에도 stop_reason=='model_error' 없음.
    (화면의 '측정됨'과 실제 재사용 가능성이 어긋나면 안 되므로 엔진과 동일 취지로 판정.
    arena.py를 임포트하지 않고 같은 기준을 여기서 자체 구현한다.)
    """
    if not isinstance(summary, dict) or "status" in summary or slug in failed_slugs:
        return False
    eps = summary.get("episodes")
    if not isinstance(eps, list) or not isinstance(episodes, int) or len(eps) != episodes:
        return False
    return not any(isinstance(e, dict) and e.get("stop_reason") == "model_error" for e in eps)


def _seed_groups(root: Path) -> dict:
    """전 런을 측정 조건 단위로 묶어 집계 — 단, 반복 수만 다른 repeat_seed 런들은 뷰에서 병합.

    병합 키: repeat_seed 런(seeds 전부 동일, [S]·[S,S,S] 포함)은 (game, seed_base, max_turns)로
    묶어 반복 수(에피소드) 무관 한 행. 구형 연속 시드([S,S+1..])·시드 없는 런은 종전대로 조건별
    분리. measured는 그룹 내 전 런의 완주 슬러그 합집합. preparing/실패 런도 runs엔 넣되 coverage엔
    미기여. **measurement_key·재사용 로직은 무변경 — 이 병합은 순수 뷰(표시) 레벨이다.**

    출력 필드: 기존(seed·seeds·game·episodes·max_turns·runs·measured·latest_run)은 최신 런 기준으로
    유지(채우기 프리필 기본=최신 조건)하고, 병합 메타를 추가한다:
      merged(런 2개 이상), plays(Σ 에피소드), episode_breakdown([{episodes,runs}] 오름차순).
    """
    index = _read_json(root / "index.json") or {"runs": []}
    groups = {}
    for run in index.get("runs", []):
        if not isinstance(run, dict):
            continue
        rid = run.get("run_id")
        if not isinstance(rid, str):
            continue
        mpath = _safe_path(root, rid, "manifest.json")
        manifest = (_read_json(mpath) if mpath else None) or {}
        seeds = manifest.get("seeds")
        seeds_list = list(seeds) if (isinstance(seeds, list) and seeds
                                     and all(_num(x) for x in seeds)) else None
        seed = seeds_list[0] if seeds_list else None
        game = manifest.get("game") or run.get("game") or "ko-semantle"
        episodes = manifest.get("episodes")
        if not _num(episodes):
            episodes = run.get("episodes")
        max_turns = manifest.get("max_turns")
        if not _num(max_turns):
            max_turns = run.get("max_turns")
        status = run.get("status") or manifest.get("status")
        started = run.get("started_at") or manifest.get("started_at")
        slugs = manifest.get("models")
        if not isinstance(slugs, list):
            slugs = [p.get("slug") for p in manifest.get("participants", []) if isinstance(p, dict)]
        failed_slugs = {f.get("slug") for f in (manifest.get("failed_participants") or [])
                        if isinstance(f, dict)}
        measured = set()
        for slug in slugs:
            if not isinstance(slug, str):
                continue
            spath = _safe_path(root, rid, "models", slug, "summary.json")
            summary = _read_json(spath) if spath else None
            if _participant_complete(summary, slug, failed_slugs, episodes):
                measured.add(slug)
        ep_key = episodes if _num(episodes) else None
        mt_key = max_turns if _num(max_turns) else None
        # 반복 수만 다른 repeat_seed 런은 (game, seed, max_turns)로 병합 — 에피소드는 키에서 제외.
        # 연속 시드([S,S+1..])는 seeds 전체를 키에 넣어 분리, 시드 없는 런도 조건별 분리(현행 유지).
        is_repeat = seeds_list is not None and all(s == seeds_list[0] for s in seeds_list)
        if is_repeat:
            key = ("rep", game, seed, mt_key)
        elif seeds_list is not None:
            key = ("seq", tuple(seeds_list), game, ep_key, mt_key)
        else:
            key = ("nil", game, ep_key, mt_key)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"seed": seed, "game": game, "max_turns": mt_key,
                               "runs": [], "measured": set()}
        g["runs"].append({"run_id": rid, "started_at": started, "status": status,
                          "episodes": ep_key, "seeds": seeds_list})
        g["measured"].update(measured)
    out = []
    for g in groups.values():
        runs = sorted(g["runs"], key=lambda r: r.get("started_at") or "", reverse=True)
        done = [r for r in runs if r.get("status") == "done"]
        latest = (done or runs)[0] if runs else None
        latest_run = latest["run_id"] if latest else None
        # 대표값 = 최신 런 조건(채우기 프리필 기본). 병합 그룹의 seeds/episodes도 최신 런 기준.
        latest_ep = latest["episodes"] if latest else None
        latest_seeds = latest["seeds"] if latest else ([g["seed"]] if g["seed"] is not None else None)
        # 반복 구성: 에피소드 수별 런 개수(라벨 "1판×1 + 4회 반복×1") + 총 플레이 수(Σ 에피소드).
        breakdown, plays = {}, 0
        for r in runs:
            e = r["episodes"] if _num(r["episodes"]) else 1
            breakdown[e] = breakdown.get(e, 0) + 1
            plays += e
        episode_breakdown = [{"episodes": e, "runs": c} for e, c in sorted(breakdown.items())]
        out.append({
            "seed": g["seed"], "seeds": latest_seeds, "game": g["game"],
            "episodes": latest_ep, "max_turns": g["max_turns"],
            "runs": [{"run_id": r["run_id"], "started_at": r["started_at"], "status": r["status"]}
                     for r in runs],
            "measured": sorted(g["measured"]), "latest_run": latest_run,
            # 신규(추가 전용): 뷰 병합 메타
            "merged": len(runs) > 1, "plays": plays, "episode_breakdown": episode_breakdown,
        })
    # 그룹 정렬: 최근 활동(그룹 내 최신 started_at) 내림차순.
    out.sort(key=lambda g: max((r.get("started_at") or "") for r in g["runs"]) if g["runs"] else "",
             reverse=True)
    return {"groups": out}


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
        if path == "/api/seeds":
            self._api_seeds()
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
    def _read_body(self):
        """요청 본문을 JSON으로. (obj, None) 또는 (None, error_message)."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            return None, "본문 길이 오류"
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except (ValueError, UnicodeDecodeError):
            return None, "JSON 파싱 실패"

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            self._post_run()
            return
        if parsed.path == "/api/stop":
            self._post_stop()
            return
        self._error(404, "not found")

    def _post_run(self):
        _reap_procs()   # 새 스폰 전 종료된 러너 회수(defunct 방지)
        body, err = self._read_body()
        if err:
            self._error(400, err)
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
            "seed": payload["seed"], "reuse": payload["reuse"],
            "repeat_seed": payload["repeat_seed"], "workers": payload["workers"],
        }, status=202)

    def _post_stop(self):
        """POST /api/stop {run_id} — 실행 중인 런을 정지(그룹 SIGTERM→SIGKILL) 후 stopped로 기록.

        멱등: 이미 done/failed/stopped면 시그널 없이 현재 status 반환. 없는 run_id는 404.
        정지 대상 pid는 manifest['pid'](서버 재시작 후에도 동작) → 없으면 보관한 Popen 보조.
        """
        _reap_procs()
        body, err = self._read_body()
        if err:
            self._error(400, err)
            return
        if not isinstance(body, dict):
            self._error(400, "본문이 JSON 객체가 아님")
            return
        run_id = body.get("run_id")
        if not isinstance(run_id, str) or not SEG_RE.match(run_id):
            self._error(400, "run_id 형식 오류")
            return
        mpath = _safe_path(self.root, run_id, "manifest.json")
        manifest = _read_json(mpath) if mpath else None
        if manifest is None:
            self._error(404, "존재하지 않는 run")
            return
        status = manifest.get("status")
        if status in ("done", "failed", "stopped"):
            # 이미 끝난 런: 멱등 — 시그널 없이 현재 status 반환.
            self._json({"ok": True, "status": status, "already": True})
            return
        pid = manifest.get("pid")
        valid_pid = isinstance(pid, int) and not isinstance(pid, bool)
        if valid_pid:
            _stop_process(pid, _PROCS.get(pid))   # 그룹 종료(best-effort)
            _PROCS.pop(pid, None)
        # 프로세스 유무와 무관하게 정지 상태를 정직히 기록(사용자 의도 반영).
        finished_at = datetime.now().replace(microsecond=0).isoformat()
        manifest["status"] = "stopped"
        manifest["finished_at"] = finished_at
        _write_json_atomic(mpath, manifest)
        _mark_index_stopped(self.root, run_id, finished_at)
        self._json({"ok": True, "status": "stopped"})

    # --- API 구현 ---
    def _api_index(self):
        _reap_procs()   # 상태 조회 시점에도 종료된 러너 회수(defunct 방지)
        data = _read_json(self.root / "index.json") or {"runs": []}
        runs = data.get("runs")
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                count, winner = _run_summary(self.root, run)
                run["participants_count"] = count
                run["winner"] = winner
        self._json(data)

    def _api_seeds(self):
        self._json(_seed_groups(self.root))

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
        # manifest 참가자 전원을 합집합에 — 디렉터리 없는(대기 중) slug는 플레이스홀더
        manifest_slugs = manifest.get("models")
        if not isinstance(manifest_slugs, list):
            manifest_slugs = [p.get("slug") for p in manifest.get("participants", []) if isinstance(p, dict)]
        for slug in manifest_slugs:
            if isinstance(slug, str) and SEG_RE.match(slug) and slug not in models_out:
                models_out[slug] = {"live": None, "summary": None, "events_count": 0}
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
            # seed_base(추가 전용): 모델별 탭이 (게임·시드·턴) 조건으로 묶어 반복/재실행을 평균하는 데 쓴다.
            mpath = _safe_path(self.root, rid, "manifest.json")
            man = (_read_json(mpath) if mpath else None) or {}
            mseeds = man.get("seeds")
            seed_base = mseeds[0] if (isinstance(mseeds, list) and mseeds and _num(mseeds[0])) else None
            runs_out.append({
                "run_id": rid,
                "game": run.get("game"),
                "status": run.get("status"),
                "episodes": run.get("episodes"),
                "max_turns": run.get("max_turns"),
                "effort": run.get("effort"),
                "seed": seed_base,
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
<meta name="viewport" content="width=device-width, initial-scale=1">
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
  --prep:#ef8f3c;   /* 준비 중(오라클 로딩): running(gold)과 구분되는 웜 오렌지 */
  --font:-apple-system,"Apple SD Gothic Neo","Pretendard",system-ui,"Segoe UI",Roboto,sans-serif;
  --lane-min:42px;
}
:root[data-theme="light"]{
  --plane:#f2f1ec; --surf:#fcfcfb; --surf2:#f6f5f0; --surf3:#eeece5;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7873;
  --grid:#e1e0d9; --axis:#c3c2b7; --line:rgba(11,11,11,.12);
  --m1:#2a78d6; --m2:#1baf7a; --m3:#4a3aa7; --m4:#d84a86;
  --m5:#0f8a2e; --m6:#e34948; --m7:#b07c00; --m8:#d1521f;
  --h0:#dd9a08; --h1:#c07f00; --h2:#9c6400; --h3:#734900; --h4:#4d3100;
  --good:#0a7d0a; --warn:#a86a00; --crit:#c8322f; --gold:#a86a00;
  --prep:#c2661a;
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

.status{display:flex;align-items:center;justify-content:center;gap:7px;padding:0 11px;height:30px;
  min-width:82px;border-radius:20px;font-size:12px;font-weight:700;letter-spacing:.03em}
.status.live{background:rgba(255,200,87,.14);color:var(--gold)}
.status.done{background:var(--surf3);color:var(--ink2)}
.status .dot{width:8px;height:8px;border-radius:50%}
.status.live .dot{background:var(--gold);animation:pulse 1.6s ease-in-out infinite}
.status.done .dot{background:var(--good)}
.status.failed{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.status.failed .dot{background:var(--warn)}
/* 정지됨(사용자 정지): 뉴트럴 뮤트 톤 — 완료와 구분되되 실패보다 중립 */
.status.stopped{background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--ink2)}
.status.stopped .dot{background:var(--muted)}
/* 준비 중: 웜 오렌지 + 맥동 도트(진행 중임을 표시, LIVE와 색으로 구분) */
.status.preparing{background:color-mix(in srgb,var(--prep) 16%,transparent);color:var(--prep)}
.status.preparing .dot{background:var(--prep);animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.7)}}

.meta{display:flex;gap:6px}
.chip{font-size:11px;color:var(--ink2);background:var(--surf2);
  border:1px solid var(--line);border-radius:7px;padding:3px 8px;white-space:nowrap}
.chip b{color:var(--ink);font-variant-numeric:tabular-nums}

.spacer{flex:1}
.actions{display:flex;gap:8px;align-items:center}
.btn{height:34px;padding:0 14px;border-radius:9px;font-size:13px;font-weight:600;white-space:nowrap;flex:0 0 auto;
  background:var(--surf2);border:1px solid var(--line)}
/* 좁은 화면: 헤더 부제·메타칩 생략해 런 선택·버튼 공간 확보(정보는 보드에 있음) */
@media (max-width:1080px){
  header{gap:10px;padding:0 12px}
  .brand .game{display:none}
  .meta{display:none}
}
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

/* ===== 레인 타이포/간격 스케일(반응형 3단 + dense) =====
   조건부 슬롯 높이(--h-*)를 폰트와 함께 스케일 → 폴링 시 레인 높이 불변 유지(직전 작업 원칙).
   좁으면 컴팩트, 넓으면 시원하게(글자·간격 확대). dense는 '행은 얇게, 글자는 가독'. */
:root{
  --f-alias:18px; --f-big:34px; --f-word:17px; --f-tn:17px; --f-hchip:12.5px; --f-cost:12.5px;
  --h-ticker:19px; --h-hist:18px; --h-cost:13px; --h-gen:14px; --h-stat:13px;
  --lane-gap:16px; --lane-pad:3px 18px 3px 18px;
  --col1:36px; --col2:224px; --col3:158px; --col5:98px;
}
.dense{                                   /* 행은 얇게(패딩·col 축소) · 글자는 가독 하한 유지 */
  --f-alias:16px; --f-big:28px; --f-word:15px; --f-tn:15px;
  --h-ticker:17px; --h-hist:16px; --h-cost:12px; --h-gen:13px; --h-stat:12px;
  --lane-gap:14px; --lane-pad:2px 16px 2px 15px;
  --col1:32px; --col2:198px; --col3:134px; --col5:82px;
}
@media (max-width:1080px){                /* 좁은 화면: 현 밀도 유지(컴팩트) */
  :root{ --f-alias:16px; --f-big:30px; --f-word:15px; --f-tn:15px; --f-hchip:12px; --f-cost:12px;
    --h-ticker:17px; --h-hist:17px; --h-cost:13px; --h-gen:13px; --h-stat:12px;
    --lane-gap:12px; --col2:196px; --col3:138px; --col5:84px; }
  .dense{ --f-alias:15px; --f-big:26px; --f-word:14px; --f-tn:14px;
    --h-ticker:16px; --h-hist:16px; --h-cost:12px; --h-gen:13px; --h-stat:12px;
    --lane-gap:10px; --col2:176px; --col3:122px; --col5:74px; }
}
@media (min-width:1500px){                /* 와이드: 글자만 확대(높이는 압축 유지 — 와이드 여백 낭비 제거) */
  :root{ --f-alias:20px; --f-big:38px; --f-word:19px; --f-tn:19px; --f-hchip:13.5px; --f-cost:13.5px;
    --h-ticker:21px; --h-hist:20px; --h-cost:15px; --h-gen:16px; --h-stat:14px;
    --lane-gap:24px; --col2:264px; --col3:180px; --col5:116px; }
  .dense{ --f-alias:17px; --f-big:31px; --f-word:16px; --f-tn:16px;
    --h-ticker:18px; --h-hist:19px; --h-cost:14px; --h-gen:15px; --h-stat:13px;
    --lane-gap:18px; --col2:220px; --col3:156px; --col5:100px; }
}

/* ---- 레인 ---- */
.lane{
  position:relative;flex:0 0 auto;min-height:var(--lane-min);
  display:grid;align-items:center;
  grid-template-columns:var(--col1) var(--col2) var(--col3) 1fr var(--col5);
  gap:var(--lane-gap);padding:var(--lane-pad);
  background:var(--surf);border:1px solid var(--line);border-radius:12px;
  overflow:hidden;transition:background .18s,border-color .25s,box-shadow .25s;
}
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
/* 대기 중(아직 시작 안 한 참가자): 뮤트 톤 + 액센트 바 채도 낮춤 */
.lane.waiting{opacity:.45}
.lane.waiting::before{filter:saturate(.5)}
/* 게임 준비 중(오라클 로딩): 대기보다 존재감 있게 + 오렌지 액센트 바가 로딩처럼 명멸 */
.lane.preparing{opacity:.8}
.lane.preparing::before{background:var(--prep);animation:shimmer 1.8s ease-in-out infinite}

/* col1: 순위 위치 */
.pos{font-size:20px;font-weight:800;color:var(--muted);text-align:center;font-variant-numeric:tabular-nums}
.lane.leader .pos{color:var(--gold)}
.dense .pos{font-size:17px}

/* col2: 정체성 */
.who{display:flex;align-items:center;gap:10px;min-width:0}
.who .keydot{width:12px;height:12px;border-radius:50%;background:var(--mc);flex:0 0 auto;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--mc) 22%,transparent)}
.who .names{min-width:0}
.who .aline{display:flex;align-items:center;gap:6px;min-width:0}
.who .alias{font-size:var(--f-alias);font-weight:750;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* effort 배지(레인·드로어·기록 공용) */
.eff{font-size:11px;font-weight:700;letter-spacing:.02em;padding:1px 6px;border-radius:5px;
  background:var(--surf3);color:var(--ink2);text-transform:uppercase;flex:0 0 auto;white-space:nowrap}
/* 재사용 배지: 저장 결과를 그대로 편입한 참가자 표식(메타 톤 — 과한 강조 금지) */
.rbadge{font-size:11px;font-weight:700;letter-spacing:.02em;padding:1px 6px 1px 5px;border-radius:5px;
  background:var(--surf3);color:var(--muted);border:1px solid var(--line);
  flex:0 0 auto;white-space:nowrap;display:inline-flex;align-items:center;gap:3px}
.rbadge::before{content:"↻";font-size:10px;line-height:1;opacity:.85}
.who .mid{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:.01em}
.dense .who .mid{display:none}
.who .phase{font-size:11.5px;font-weight:700;letter-spacing:.04em;padding:1px 6px;border-radius:5px;margin-top:2px;display:inline-block}
.phase.running{color:var(--gold);background:rgba(255,200,87,.13)}
.phase.done{color:var(--muted);background:var(--surf3)}
.phase.solvedp{color:var(--good);background:color-mix(in srgb,var(--good) 16%,transparent)}
.phase.waiting{color:var(--muted);background:var(--surf3)}
.phase.preparing{color:var(--prep);background:color-mix(in srgb,var(--prep) 15%,transparent)}
.phase.aborted{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
/* 턴 소진 실패(완주·미해결): 회색-붉은 톤 — 진행 중과 확연히 구분(강등된 실패) */
.phase.exhausted{color:var(--crit);background:color-mix(in srgb,var(--crit) 13%,transparent)}
.dense .who .phase{display:none}

/* col3: 헤드라인 순위 */
.rankcell{text-align:right}
.rankcell .cap{font-size:11.5px;font-weight:700;letter-spacing:.04em;color:var(--muted);margin-bottom:1px}
.rankcell .big{font-size:var(--f-big);font-weight:800;line-height:1;font-variant-numeric:tabular-nums;
  color:var(--hc);transition:color .3s;display:inline-block}
.rankcell .big.flash{animation:rankpop .5s ease}
@keyframes rankpop{0%{transform:scale(1)}30%{transform:scale(1.14)}100%{transform:scale(1)}}
.rankcell .unit{font-size:14px;font-weight:600;color:var(--muted);margin-left:1px}
.rankcell .denom{font-size:13px;font-weight:600;color:var(--muted);margin-left:3px}
.rankcell .big.na{color:var(--muted);font-size:calc(var(--f-big) * .7)}
.rankcell .stat-upd{font-size:12px;font-weight:700;margin-top:3px;letter-spacing:.02em;
  min-height:var(--h-stat);line-height:var(--h-stat)}
.rankcell .stat-upd.fresh{color:var(--good)}
.rankcell .stat-upd.stall{color:var(--muted)}
.rankcell .stat-upd.stall.long{color:var(--warn)}
.rankcell .stat-upd.solvebasis{color:var(--good);font-weight:800}

/* col4: 생성문 · 티커 · 기록 */
.mid-col{display:flex;flex-direction:column;gap:4px;min-width:0}
.ticker{display:flex;align-items:center;gap:9px;min-width:0;font-size:13px;
  min-height:var(--h-ticker);line-height:var(--h-ticker)}
.ticker .verb{font-size:11px;color:var(--muted);letter-spacing:.05em;flex:0 0 auto}
.ticker .word{font-weight:750;font-size:var(--f-word);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
.ticker .sim{color:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums;flex:0 0 auto}
.ticker .sim b{color:var(--ink2);font-weight:600}
:root[data-theme="light"] .ticker .sim b{color:var(--ink2)}
.ticker .rk{color:var(--ink);font-weight:700;font-variant-numeric:tabular-nums;flex:0 0 auto}
.ticker.flash .word{animation:tflash .7s ease}
@keyframes tflash{0%{color:var(--gold)}100%{color:var(--ink)}}
.ticker .bad{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:6px;flex:0 0 auto;line-height:1}
.ticker .bad.fmt{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.ticker .bad.dup{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.ticker .target{margin-left:auto;font-size:12.5px;color:var(--good);font-weight:700;flex:0 0 auto}
.ticker .target b{font-size:14px}
/* 티커 자식 라인박스를 작게 고정 → 내용(정답/추측/무효)과 무관하게 슬롯 min-height가 높이를 지배(균일) */
.ticker span,.ticker b{line-height:1.1}
.dense .ticker .verb{display:none}

/* col5: 턴 */
.turns{text-align:right}
.turns .tn{font-size:var(--f-tn);font-weight:700;font-variant-numeric:tabular-nums}
.turns .tn .sep{color:var(--muted);font-weight:500}
.turns .lab{font-size:11.5px;color:var(--muted);letter-spacing:.04em}
.turns .inv{font-size:11.5px;color:var(--warn);margin-top:2px;font-variant-numeric:tabular-nums}
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
/* 벤더 그룹 [전체] 토글: 그룹 전 모델 선택/해제 */
.fam-toggle{margin-left:auto;align-self:center;font-size:10px;font-weight:700;padding:2px 9px;border-radius:6px;
  background:var(--surf2);border:1px solid var(--line);color:var(--muted);cursor:pointer;
  letter-spacing:normal;text-transform:none;line-height:1.5}
.fam-toggle:hover{border-color:var(--axis);color:var(--ink2)}
.fam-toggle.on{background:var(--gold);color:#1a1200;border-color:transparent}
/* 참가자 헤더: 라벨 + 일괄 버튼(좁은 화면에서 줄바꿈) */
.selhead{display:flex;align-items:center;flex-wrap:wrap;gap:6px 10px;justify-content:space-between}
.selhead-lab{flex:1 1 auto}
.selhead-btns{display:flex;gap:6px;flex:0 0 auto}
.btn.tiny{height:26px;padding:0 10px;font-size:11.5px;font-weight:600;letter-spacing:normal;text-transform:none}
/* 일괄 effort 칩 줄: 개별 모델 칩(.echip)과 동일 모양으로 의미 동일성 시각화 */
.bulkeff{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:9px 0 3px}
.bulkeff-lab{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-right:3px}
.bulkeff .echip{cursor:pointer}
.bulkeff-msg{font-size:11px;color:var(--muted);margin-left:4px}
.mopt{display:flex;align-items:center;gap:9px;padding:6px 9px;border-radius:9px;align-self:start;
  background:var(--surf2);border:1px solid var(--line);user-select:none}
.mopt.on{border-color:var(--gold);background:color-mix(in srgb,var(--gold) 12%,transparent)}
.mopt .al{flex:0 0 auto;font-weight:750;font-size:13.5px;min-width:26px;cursor:pointer}
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

/* ---- 런처: 시드 입력(문제 세트) + 랜덤 주사위 + 재측정 토글 ---- */
.seedrow{display:flex;gap:8px;align-items:center}
.seedrow input[type=number]{flex:1}
.seeddice{flex:0 0 auto;width:44px;height:38px;padding:0;font-size:18px;line-height:1;
  background:var(--surf2);border:1px solid var(--line);border-radius:9px}
.seeddice:hover{background:var(--surf3)}
.seedhint{display:block;font-size:11.5px;color:var(--muted);line-height:1.4;margin-top:7px;margin-bottom:10px}
.optrow{display:flex;align-items:flex-start;gap:10px;padding:9px 11px;border-radius:9px;
  background:var(--surf2);border:1px solid var(--line);cursor:pointer;margin-bottom:8px}
.optrow:last-child{margin-bottom:0}
.optrow.on{border-color:color-mix(in srgb,var(--gold) 45%,transparent)}
.optrow input[type=checkbox]{appearance:none;-webkit-appearance:none;flex:0 0 auto;
  width:18px;height:18px;margin-top:1px;border-radius:5px;border:1.5px solid var(--axis);
  background:var(--surf);cursor:pointer;position:relative;transition:background .15s,border-color .15s}
.optrow input[type=checkbox]:checked{background:var(--gold);border-color:transparent}
.optrow input[type=checkbox]:checked::after{content:"✓";position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#1a1200}
.optmain{display:flex;flex-direction:column;gap:2px;min-width:0}
.opttitle{font-size:13px;font-weight:600;color:var(--ink)}
.optsub{font-size:11.5px;color:var(--muted);line-height:1.4}

/* 기록 탭 */
.tabs{display:flex;gap:6px;margin-bottom:14px}
.tab{padding:7px 14px;border-radius:9px;font-size:13px;font-weight:600;background:var(--surf2);border:1px solid var(--line);color:var(--ink2)}
.tab.on{background:var(--surf3);color:var(--ink);border-color:var(--gold)}
.mrunrow{display:flex;align-items:center;gap:12px;padding:11px 12px;border-radius:10px;
  border:1px solid var(--line);margin-bottom:8px;cursor:pointer;background:var(--surf2)}
.mrunrow:hover{background:var(--surf3)}
/* 런별 카드: 모델 나열 대신 무슨 게임·누가 1등인지 요약.
   런ID·게임 배지·상태칩은 nowrap으로 절대 안 꺾이고, 긴 이름/업적만 ellipsis. */
.runcard{display:flex;flex-direction:column;gap:6px;padding:11px 13px;border-radius:11px;
  border:1px solid var(--line);margin-bottom:8px;cursor:pointer;background:var(--surf2)}
.runcard:hover{background:var(--surf3)}
.rc-head{display:flex;align-items:center;gap:9px;min-width:0}
.rc-rid{flex:0 0 auto;font-weight:700;font-size:13px;white-space:nowrap;letter-spacing:-.01em;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.rc-game{flex:0 0 auto;white-space:nowrap;font-size:11px;font-weight:700;padding:2px 9px;
  border-radius:6px;background:var(--surf3);color:var(--ink2);border:1px solid var(--line)}
.rc-st{flex:0 0 auto;margin-left:auto;white-space:nowrap;font-size:11px;font-weight:700;
  padding:3px 10px;border-radius:20px;letter-spacing:.03em}
.st.running{color:var(--gold);background:rgba(255,200,87,.14)}
.st.preparing{color:var(--prep);background:color-mix(in srgb,var(--prep) 15%,transparent)}
.st.done{color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.st.failed{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.st.stopped{color:var(--ink2);background:color-mix(in srgb,var(--muted) 18%,transparent)}
.rc-win{display:flex;align-items:center;gap:6px;min-width:0;font-size:13px}
.rc-crown{flex:0 0 auto;font-size:10px;font-weight:800;letter-spacing:.03em;color:var(--gold);
  background:color-mix(in srgb,var(--gold) 14%,transparent);border-radius:5px;padding:2px 7px}
.rc-wname{flex:0 1 auto;min-width:0;font-weight:750;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rc-ach{flex:0 1 auto;min-width:0;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rc-ach::before{content:"— ";color:var(--muted)}
.rc-none{flex:0 1 auto;font-size:12.5px;color:var(--muted);font-weight:600}
.rc-meta{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.mrunrow .metrics{margin-left:auto;display:flex;gap:14px;text-align:right}
.mrunrow .metrics .v{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
.mrunrow .metrics .k{font-size:10px;color:var(--muted)}
.selfield{margin-bottom:14px}

/* ---- 기록: 시드별 뷰(시드 우산 → 게임(조건) 행 · coverage) ---- */
.hint-load{color:var(--muted);font-size:12px;padding:6px 2px}
.seedgrp{margin-bottom:14px;border:1px solid var(--line);border-radius:12px;background:var(--surf2);overflow:hidden}
.sg-head{display:flex;align-items:center;gap:8px;padding:9px 13px;background:var(--surf3);border-bottom:1px solid var(--line)}
.sg-seed{font-weight:800;font-size:13px;letter-spacing:.01em;font-variant-numeric:tabular-nums}
.sg-tag{font-size:10px;font-weight:700;letter-spacing:.02em;padding:1px 7px;border-radius:5px;
  color:#1a1200;background:var(--gold)}
.sg-row{border-top:1px solid var(--line)}
.sg-row:first-of-type{border-top:none}
.sgr-top{display:flex;align-items:center;gap:9px;padding:9px 13px;cursor:pointer}
.sg-row:hover>.sgr-top{background:var(--surf3)}
.sgr-game{font-weight:700;font-size:13px;flex:0 0 auto}
.sgr-game.muted{color:var(--muted);font-weight:600}
.sgr-cond{font-size:11px;color:var(--muted);flex:0 0 auto;white-space:nowrap}
.sgr-cov{font-size:11.5px;color:var(--ink2);margin-left:auto;white-space:nowrap;font-variant-numeric:tabular-nums}
.sgr-cov b{color:var(--good)}
.sgr-btns{display:flex;gap:6px;flex:0 0 auto}
.minibtn{height:26px;padding:0 10px;border-radius:7px;font-size:11px;font-weight:700;
  background:var(--surf2);border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
.minibtn:hover{background:var(--surf);color:var(--ink)}
.minibtn.fill{color:#1a1200;background:var(--gold);border-color:transparent}
.minibtn.fill:hover{filter:brightness(1.06)}
.minibtn.start{color:var(--prep);border-color:color-mix(in srgb,var(--prep) 45%,transparent)}
.minibtn.stop{color:var(--crit);border-color:color-mix(in srgb,var(--crit) 45%,transparent)}
.minibtn.stop:hover{background:color-mix(in srgb,var(--crit) 14%,transparent);color:var(--crit)}
.sgr-det{display:none;padding:2px 13px 12px;flex-direction:column;gap:8px}
.sg-row.open>.sgr-det{display:flex}
.sgr-chips{display:flex;flex-wrap:wrap;align-items:center;gap:5px}
.sgr-lab{font-size:10px;font-weight:700;letter-spacing:.04em;color:var(--muted);text-transform:uppercase;margin-right:3px}
.mchip{font-size:10.5px;font-weight:600;padding:2px 8px;border-radius:6px;
  background:color-mix(in srgb,var(--good) 14%,transparent);color:var(--good);border:1px solid color-mix(in srgb,var(--good) 30%,transparent)}
.uchip{font-size:10.5px;font-weight:600;padding:2px 8px;border-radius:6px;
  background:var(--surf3);color:var(--muted);border:1px dashed var(--axis)}
.sgr-runs{display:flex;flex-direction:column;gap:4px;margin-top:2px}
.sgr-run{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:7px;background:var(--surf);
  border:1px solid var(--line);cursor:pointer}
.sgr-run:hover{background:var(--surf3)}
.sgr-run .rid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;font-weight:700}
.sgr-run .when{margin-left:auto;font-size:10.5px;color:var(--muted);font-variant-numeric:tabular-nums}

/* 빈 상태 */
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;color:var(--muted)}
.empty .big{font-size:19px;color:var(--ink2);font-weight:650}
.empty .sub{font-size:13px}

/* ---- 생성 중(라이브 공개 출력) 라인 + 이력 스트립(레인) ---- */
/* 레이아웃 시프트 방지: 조건부 슬롯은 빈 상태에도 min-height로 자리 유지(폴링 출렁임 제거) */
.genline{display:flex;align-items:center;gap:7px;min-width:0;font-size:12px;min-height:var(--h-gen)}
.genline .genlab{flex:0 0 auto;font-size:10.5px;font-weight:700;letter-spacing:.04em;line-height:1;
  color:var(--gold);background:rgba(255,200,87,.13);border-radius:5px;padding:1px 6px;
  animation:pulse 1.6s ease-in-out infinite}
:root[data-theme="light"] .genline .genlab{background:color-mix(in srgb,var(--gold) 14%,transparent)}
.genline.waiting .genlab{color:var(--muted);background:var(--surf3)}
.genline .gentail{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--ink2);letter-spacing:-.01em;
  line-height:var(--h-gen)}   /* 스트림 유무로 높이가 흔들리지 않게 라인박스 고정(= genline min-height) */
.genline.waiting .gentail{display:none}
.cursor{flex:0 0 auto;display:inline-block;width:6px;height:1em;background:var(--gold);
  vertical-align:text-bottom;animation:blink 1s step-end infinite}
.genline.waiting .cursor{display:none}
@keyframes blink{0%,50%{opacity:1}50.01%,100%{opacity:0}}

.histstrip{display:flex;align-items:center;gap:6px;min-width:0;overflow:hidden;min-height:var(--h-hist)}
.histstrip .histlab{flex:0 0 auto;font-size:10.5px;color:var(--muted);letter-spacing:.03em}
.hchip{flex:0 0 auto;display:inline-flex;align-items:center;gap:4px;max-width:150px;
  font-size:var(--f-hchip);line-height:1;padding:3px 8px;border-radius:6px}
.hchip .hw{min-width:0;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hchip .hr{flex:0 0 auto;font-size:11px;font-weight:600;opacity:.82;font-variant-numeric:tabular-nums}
.hchip.inv{background:var(--surf3);color:var(--muted);font-weight:700;padding:3px 7px}

/* 자원 지표(시간·출력 토큰): 정렬 꼬리와 동일 축을 노출 — 기존 지표를 밀어내지 않음 */
.costline{display:flex;align-items:center;gap:12px;min-width:0;font-size:var(--f-cost);
  color:var(--muted);font-variant-numeric:tabular-nums;letter-spacing:.01em;
  min-height:var(--h-cost);line-height:var(--h-cost)}
.costline .cchip{flex:0 0 auto;white-space:nowrap}
.costline .cchip b{color:var(--ink2);font-weight:650}
.costline .cchip.cost b{color:var(--ink)}   /* 비용은 약간 더 또렷하게 */
.dense .costline{gap:9px}

/* dense(기본): 이력 숨김, 생성줄은 최소 인디케이터만 */
.dense .histstrip{display:none}
.dense .genline .gentail,.dense .genline .cursor{display:none}
.dense .genline .genlab{font-size:10px;padding:1px 6px}
/* 와이드에선 dense도 이력 칩 노출(빈 중앙~우측 활용) — display:none보다 소스 뒤에 와야 이김 */
@media (min-width:1500px){ .dense .histstrip{display:flex} }
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

/* ===================== 멀티게임 요소 ===================== */
/* ---- 런처: 게임 선택 세그먼트 + 설명 ---- */
.gameseg{display:flex;gap:6px;flex-wrap:wrap}
.gseg{flex:1 1 auto;min-width:120px;display:flex;flex-direction:column;gap:1px;
  padding:8px 11px;border-radius:9px;background:var(--surf2);border:1px solid var(--line);
  text-align:left;transition:border-color .15s,background .15s}
.gseg:hover{background:var(--surf3)}
.gseg.on{border-color:var(--gold);background:color-mix(in srgb,var(--gold) 12%,transparent)}
.gseg .gk{font-size:13.5px;font-weight:750}
.gseg.on .gk{color:var(--gold)}
.gseg .gid{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}
.gamedesc{margin-top:8px;font-size:12px;color:var(--ink2);line-height:1.4;min-height:17px}
/* ---- 채우기 모드: 측정 조건 잠금 배너 + 잠긴 필드 시각화 ---- */
.fillbanner{display:flex;align-items:center;gap:10px;margin-bottom:14px;padding:9px 12px;
  border-radius:9px;background:color-mix(in srgb,var(--gold) 12%,transparent);
  border:1px solid color-mix(in srgb,var(--gold) 42%,transparent)}
.fillbanner[hidden]{display:none}
.fb-txt{flex:1;font-size:12px;line-height:1.45;color:var(--ink2)}
.fb-unlock{flex:0 0 auto;font-size:11.5px;padding:5px 10px;white-space:nowrap}
/* 잠긴 측정 조건 입력(참가자 선택 제외): disabled → 흐리게 + 클릭 차단 */
.pb input:disabled,.pb .btn:disabled{opacity:.5;cursor:not-allowed}
.gameseg.locked{pointer-events:none}
.gameseg.locked .gseg:not(.on){opacity:.4}
.optrow.locked{opacity:.5;pointer-events:none}

/* ---- col3: 게임 지표(거리/실험/정답) ---- */
.gmetrics{text-align:right;display:flex;flex-direction:column;gap:2px;align-items:flex-end}
.gm-cap{font-size:10px;font-weight:700;letter-spacing:.04em;color:var(--muted)}
.gm-big{font-size:30px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;color:var(--ink)}
.gm-big.na{color:var(--muted);font-size:22px}
.gm-big .gm-unit{font-size:12px;font-weight:600;color:var(--muted);margin-left:2px}
.gm-big.goal{color:var(--good)}
.dense .gm-big{font-size:24px}
.gm-sub{display:flex;gap:5px;margin-top:3px;flex-wrap:wrap;justify-content:flex-end}
.gm-chip{font-size:10.5px;color:var(--ink2);background:var(--surf2);border:1px solid var(--line);
  border-radius:6px;padding:2px 7px;white-space:nowrap;font-variant-numeric:tabular-nums}
.gm-chip b{color:var(--ink);font-weight:700}
.gm-chip.warnv b{color:var(--warn)}
.dense .gm-sub{display:none}

/* ---- maze: 미니맵 + 이동 티커 ---- */
.mm-wrap{display:flex;align-items:center;gap:11px;min-width:0}
.minimap{flex:0 0 auto;border-radius:5px;background:var(--plane);
  box-shadow:inset 0 0 0 1px var(--line)}
.mm-bg{fill:var(--plane)}
.mm-cell{fill:var(--surf3)}
.mm-open{stroke:var(--mc);stroke-width:1.2;stroke-linecap:round;opacity:.75}
.mm-cur{stroke:var(--plane);stroke-width:1}
.mm-goal{fill:color-mix(in srgb,var(--good) 34%,transparent);stroke:var(--gold);stroke-width:1.3}
.maze-tick{min-width:0;font-size:12px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.maze-tick .verb{font-size:10px;color:var(--muted);letter-spacing:.05em;margin-right:5px}
.maze-tick b{font-weight:750;color:var(--ink);font-size:14px}
.maze-tick .brg{color:var(--muted);font-size:11px;margin-left:5px}
.mm-cap{font-size:9px;color:var(--muted);letter-spacing:.02em;margin-top:2px;text-align:center}
.dense .maze-tick{display:none}
.dense .mm-cap{display:none}

/* ---- rulelab: 실험 로그 ---- */
.explog{display:flex;align-items:center;gap:5px;flex-wrap:wrap;min-width:0;overflow:hidden}
.explog .verb{font-size:11px;color:var(--muted)}
.exp-chip{flex:0 0 auto;font-size:11px;line-height:1;padding:3px 8px;border-radius:6px;
  background:var(--surf2);border:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}
.exp-chip b{color:var(--ink);font-weight:700}
.exp-chip .out{color:var(--h3);font-weight:700}
:root[data-theme="light"] .exp-chip .out{color:var(--h2)}
.exp-ans{flex:0 0 auto;font-size:11px;font-weight:700;padding:3px 8px;border-radius:6px;
  color:var(--warn);background:color-mix(in srgb,var(--warn) 14%,transparent)}
.exp-ans.ok{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.dense .exp-chip:nth-child(n+4){display:none}

/* ---- minefield: 목숨 + 지뢰 배지 ---- */
.lives{display:flex;align-items:center;gap:5px;font-size:12px}
.lives .lv-lab{font-size:10px;color:var(--muted);letter-spacing:.04em;margin-right:2px}
.lives .pip{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
.lives .pip.on{background:var(--crit);box-shadow:0 0 0 2px color-mix(in srgb,var(--crit) 24%,transparent)}
.lives .pip.off{background:var(--surf3);border:1px solid var(--line)}
.lives .lv-out{font-size:10px;font-weight:800;color:var(--crit);letter-spacing:.05em;margin-left:3px}
.ticker .bad.boom{color:#fff;background:var(--crit)}
.ticker .bad.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 18%,transparent)}
.hchip.boom{background:var(--crit);color:#fff;font-weight:800;padding:2px 6px}
.hchip.warn{box-shadow:0 0 0 1.5px var(--warn)}
.lane.mined{opacity:.7}
.lane.mined::before{background:var(--crit)}
.dense .lives .lv-lab{display:none}
.mm-col{display:flex;flex-direction:column;align-items:center;flex:0 0 auto}

/* ---- rulelab 트렌드: 참가자별 실험/정답 막대 ---- */
.rl-bars{display:flex;flex-direction:column;gap:8px;height:100%;overflow-y:auto;padding:4px 2px}
.rl-row{display:grid;grid-template-columns:150px 1fr 118px;gap:12px;align-items:center}
.rl-nm{display:flex;align-items:center;gap:7px;min-width:0;font-size:12px;color:var(--ink2)}
.rl-nm .rl-sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
.rl-nm .rl-nmt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rl-barwrap{position:relative;height:20px;background:var(--surf2);border:1px solid var(--line);border-radius:6px;overflow:hidden;display:flex;align-items:center}
.rl-bar{height:100%;border-radius:0;opacity:.55;min-width:2px}
.rl-bl{position:absolute;left:8px;font-size:11px;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;white-space:nowrap}
.rl-correct{display:flex;align-items:center;gap:4px;justify-content:flex-end}
.rl-pip{width:12px;height:12px;border-radius:3px;background:var(--surf3);border:1px solid var(--line);flex:0 0 auto}
.rl-pip.on{background:var(--good);border-color:transparent}
.rl-cn{font-size:12px;font-weight:750;font-variant-numeric:tabular-nums;margin-left:4px}
.rl-cn.muted{color:var(--muted);font-weight:600}

.hidden{display:none!important}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="brand">
      <span class="mark">MIND<b>MATCH</b></span>
      <span class="game" id="brandGame"><span class="kr">꼬맨틀</span> · 동시 관전</span>
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
      <!-- 채우기 모드 배너: 측정 조건 잠금 안내 + 조건 잠금 해제(일반 모드 전환) -->
      <div class="fillbanner" id="fillBanner" hidden>
        <span class="fb-txt" id="fillBannerText"></span>
        <button type="button" class="btn fb-unlock" id="fillUnlockBtn">조건 잠금 해제</button>
      </div>
      <div class="field">
        <span class="k">게임 선택</span>
        <div class="gameseg" id="gameSeg"></div>
        <div class="gamedesc" id="gameDesc"></div>
      </div>
      <div class="field">
        <span class="k selhead">
          <span class="selhead-lab">참가자 선택 · effort 칩을 눌러 추가 <span class="count" id="mcount"></span></span>
          <span class="selhead-btns">
            <button type="button" class="btn tiny" id="selNoneBtn">모두 해제</button>
          </span>
        </span>
        <!-- 전 모델 effort 토글: 켜기 = 지원하는 전 모델에 그 effort 추가(기존 선택 보존), 끄기 = 전 모델에서 제거 -->
        <div class="bulkeff" id="bulkEff"><span class="bulkeff-lab">전 모델 effort</span><span class="bulkeff-msg" id="bulkEffMsg"></span></div>
        <div class="modelgrid" id="modelGrid"></div>
      </div>
      <div class="inline">
        <div class="field"><span class="k">반복 수</span><input type="number" id="fEpisodes" min="1" max="50" value="1"></div>
        <div class="field"><span class="k">최대 턴</span><input type="number" id="fTurns" min="1" max="200" value="15"></div>
        <div class="field"><span class="k">최대 동시 실행</span><input type="number" id="fWorkers" min="1" max="128" value="32"></div>
      </div>
      <span class="seedhint">반복 수 = 같은 문제(시드)를 N회 풀어 안정성을 봅니다. 1회면 단판입니다.</span>
      <span class="seedhint">최대 동시 실행 = 한 번에 보내는 플레이 수. 참가자가 더 많으면 나머지는 차례를 기다립니다(자원 보호).</span>
      <div class="field">
        <span class="k">시드(문제 세트)</span>
        <div class="seedrow">
          <input type="number" id="fSeed" min="0" step="1" value="314159">
          <button type="button" class="btn seeddice" id="seedDice" title="랜덤 시드 생성">🎲</button>
        </div>
        <span class="seedhint">같은 시드·판수·턴수 = 같은 문제 세트 → 완주한 결과는 자동 재활용. 새 문제를 원하면 🎲로 시드를 바꾸세요.</span>
        <label class="optrow" id="remeasureRow" for="fRemeasure">
          <input type="checkbox" id="fRemeasure">
          <span class="optmain">
            <span class="opttitle">전부 다시 측정 — 재사용 없이 모든 참가자를 새로 실행</span>
          </span>
        </label>
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
        <button class="tab on" id="tabSeeds">시드별</button>
        <button class="tab" id="tabRuns">날짜별</button>
        <button class="tab" id="tabModels">모델별</button>
      </div>
      <div id="histSeeds"></div>
      <div id="histRuns" class="hidden"></div>
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
    let model=slug, effort=man.effort||'low', reusedFrom=null;
    if(plist){ const p=plist.find(x=>x&&x.slug===slug); if(p){ model=p.model||model; effort=p.effort||effort; if(p.reused_from) reusedFrom=p.reused_from; } }
    else if(slug.includes('@')){ const a=slug.split('@'); model=a[0]; effort=a[1]||effort; }
    const live=S.run.models[slug].live;   // v2 live/summary는 effort·순수 model 포함
    if(live){ if(live.model) model=live.model; if(live.effort) effort=live.effort; }
    S.parts[slug]={slug, model, effort, alias:modelAlias(model), name:nameOf(model), reusedFrom};
  });
}
function modelAlias(m){const f=CFG.models.find(x=>x.id===m);return f?f.alias:m;}
function nameOf(m){const f=CFG.models.find(x=>x.id===m);return (f&&f.name)?f.name:(f?f.alias:m);}
function aliasOf(slug){ return (S.parts[slug]&&S.parts[slug].name) || nameOf(String(slug).split('@')[0]); }
function effortOf(slug){ return (S.parts[slug]&&S.parts[slug].effort) || (String(slug).includes('@')?slug.split('@')[1]:''); }
function slugLabel(slug){ // 컨텍스트에 참가자 메타가 없을 때(기록 등): "풀네임 · effort"
  const a=String(slug).split('@'); return a.length>1? nameOf(a[0])+' · '+a[1] : nameOf(a[0]);
}
function reusedFromOf(slug){ return (S.parts[slug]&&S.parts[slug].reusedFrom)||null; }
function reusedBadge(slug){ // 재사용 참가자(저장 결과 편입)만 메타 톤 배지 + 출처 run_id 툴팁
  const rf=reusedFromOf(slug); if(!rf) return null;
  const b=el('span','rbadge','재사용');
  b.title='재사용: '+String(rf).replace(/^arena-/,'')+' 결과 편입';
  return b;
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
  const man=S.run.manifest||{};
  if(man.status==='done'||man.status==='failed'||man.status==='stopped'||man.finished_at) return false;   // 종료(정지 포함) 우선 — 고아 레인 phase 무시
  if(man.status==='preparing') return true;   // 준비 중(예비 manifest) — 폴링 유지, running으로 자연 전환
  if(man.status==='running') return true;
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
  // 스트림 없음 → display:none(레이아웃 붕괴)이 아니라 visibility:hidden으로 '자리'는 유지(높이 불변).
  if(!st.show){ gl.style.visibility='hidden'; return; }
  gl.style.visibility='visible';
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
  const man=S.run.manifest||{};
  const runOver=(man.status==='done'||man.status==='failed'||!!man.finished_at);
  let phase='done';
  if(end){ phase = end.solved ? 'solved' : 'done'; }
  else if(live){
    if(live.error || live.phase==='failed'){ phase='aborted'; }
    else if(live.phase==='running'){ phase = runOver ? 'aborted' : 'running'; }   // 종료된 런의 running = 고아
    else { phase='done'; }                                                        // unknown phase → done
  }
  else if(turns.length===0){ phase='waiting'; }   // 아직 시작 안 한 대기 참가자
  else { phase='done'; }
  // 예비 manifest(status:preparing) 단계: 아직 데이터 없는 레인은 '게임 준비 중'으로 표시
  if(man.status==='preparing' && !end && turns.length===0){ phase='preparing'; }
  const lastValid=[...valid].reverse()[0]||null;
  const cost=usageOf(turns);
  return {
    episode:ep, turns, valid, last, end, bestRank, phase,
    solved: !!(end&&end.solved),
    target: end?end.target:null,     // 서버가 이미 진행 중 에피소드 target을 뺌
    maxTurns:(live&&live.max_turns)|| (S.run.manifest&&S.run.manifest.max_turns) || null,
    invalidCount: turns.filter(t=>!t.valid).length,
    turnsUsed: last?last.turn:(end?end.turns:0),   // 정렬 꼬리 ①: 사용 턴 수
    durMs: cost.durMs, outTok: cost.outTok, costUsd: cost.costUsd,   // 정렬 꼬리 ②③ + 비용(결측 null)
    lastSim: last&&last.valid?last.similarity:(lastValid?lastValid.similarity:(live&&isCur?live.last_similarity:null)),
  };
}
/* 턴 이벤트의 usage 합(구형 런엔 usage 없음 → 해당 지표 null 유지). */
function usageOf(turns){
  let durMs=null, outTok=null, costUsd=null;
  for(const t of turns){
    const u=t&&t.usage; if(!u) continue;
    if(typeof u.duration_ms==='number') durMs=(durMs==null?0:durMs)+u.duration_ms;
    if(typeof u.output_tokens==='number') outTok=(outTok==null?0:outTok)+u.output_tokens;
    if(typeof u.cost_usd==='number') costUsd=(costUsd==null?0:costUsd)+u.cost_usd;   // 턴 usage 누적(live도 갱신)
  }
  return {durMs, outTok, costUsd};
}
// 정렬 티어: 정답(0) > 진행 중(1) > 턴 소진·중단 실패(2) > 대기·준비(3).
// 맞히지 못하고 턴을 다 쓴(또는 중단된) 참가자는 진행도가 좋아도 진행 중 아래로 강등.
function sortTier(v){
  if(v.solved) return 0;
  if(v.phase==='running') return 1;
  if(v.phase==='waiting'||v.phase==='preparing') return 3;
  return 2;   // done/aborted 미해결 = 완주·미해결(턴 소진) 또는 중단
}
// 턴 소진 실패(완주했으나 미해결) — 중단(aborted 고아)과는 구분해 'N턴 소진'으로 명시.
function isExhausted(v){ return v.phase==='done' && !v.solved; }
// 로드된 이벤트 전체에 usage(시간·토큰)가 하나라도 있나. 없으면 캡션이 시간·토큰
// 순을 약속하지 않도록(구형 런 = usage 도입 전 → 시간·토큰 기록 없음).
function runHasUsage(){
  for(const m in S.run.models){
    const slot=S.ev[m]; if(!slot) continue;
    for(const e of slot.events){
      if(e && e.type==='turn' && e.usage &&
         (typeof e.usage.duration_ms==='number' || typeof e.usage.output_tokens==='number')) return true;
    }
  }
  return false;
}
// 정답 레인 순위 근거: 정렬 꼬리(turnsUsed)와 '같은 값'으로 '정답까지 N턴' 표기(표시=근거 일치).
function solveBasis(v){
  const s=el('div','stat-upd solvebasis');
  s.textContent='정답까지 '+(v.turnsUsed||0)+'턴';
  return s;
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

// seeds 판독(단일 소스): 반복(전부 동일·길이>1)=시도, 구형 다문제(서로 다름)=에피소드, 그 외=단판.
function seedsInfo(man){
  const s=man&&man.seeds;
  if(Array.isArray(s)&&s.length>1){
    return s.every(x=>x===s[0]) ? {kind:'repeat', n:s.length, label:'시도'}
                                : {kind:'multi', n:s.length, label:'에피소드'};
  }
  return {kind:'single', n:(Array.isArray(s)?s.length:1), label:'에피소드'};
}
// 시드별 기록 그룹의 조건 라벨: 반복='N회 반복×T턴', 구형 다문제='N문제(연속 시드)×T턴', 단판='1판×T턴'.
// 한 런의 반복 횟수(에피소드) → 사람 표기('1판' / 'N회 반복').
function playLabel(ep){ return (ep>1) ? (ep+'회 반복') : '1판'; }
function seedCondLabel(g){
  const s=g.seeds, mt=g.max_turns||'?', ep=g.episodes||1;
  // 뷰 병합 그룹(반복 수만 다른 런들 합침): "N턴 · P플레이(1판×1 + 4회 반복×1)"로 구성 정직 표기.
  if(g.merged && Array.isArray(g.episode_breakdown) && g.episode_breakdown.length){
    const parts=g.episode_breakdown.map(b=>playLabel(b.episodes)+'×'+b.runs);
    return mt+'턴 · '+(g.plays||0)+'플레이('+parts.join(' + ')+')';
  }
  if(Array.isArray(s)&&s.length>1){
    return s.every(x=>x===s[0]) ? (s.length+'회 반복×'+mt+'턴')
                                : (s.length+'문제(연속 시드)×'+mt+'턴');
  }
  if(!Array.isArray(s)&&ep>1) return ep+'에피소드×'+mt+'턴';   // 시드 없는 구형 다에피소드
  return '1판×'+mt+'턴';
}

function renderTopbar(){
  if(!S.runId||!S.run){return;}
  const row=((S.index&&S.index.runs)||[]).find(r=>r.run_id===S.runId)||{};
  const man=(S.run&&S.run.manifest)||{};
  const models=Object.keys(S.run.models||{});
  const bg=$('#brandGame');
  if(bg){ const kr=(gmeta(man.game||'ko-semantle').kr)||man.game||'꼬맨틀';
    bg.innerHTML='<span class="kr">'+esc(kr)+'</span> · 동시 관전'; }
  $('#runselRid').textContent=S.runId.replace(/^arena-/,'');
  $('#runselSub').textContent=`${models.length} 참가자`;
  const live=isLive();
  const st=$('#status');
  // 준비 중은 live(폴링용)로 잡히므로 live 분기보다 먼저 판정(LIVE 오표기 방지).
  if(man.status==='preparing'){ st.className='status preparing'; $('#statusTxt').textContent='준비 중'; st.title='게임 준비 중 — 오라클 로딩(임베딩 모델·기준 어휘)'; }
  else if(live){ st.className='status live'; $('#statusTxt').textContent='LIVE'; st.title=''; }
  else if(man.status==='failed'||man.failure){ st.className='status failed'; $('#statusTxt').textContent='중단'; const reason=man.failure||man.error; st.title=reason?('중단: '+reason):'실행 중단됨'; }
  else if(man.status==='stopped'){ st.className='status stopped'; $('#statusTxt').textContent='정지됨'; st.title='사용자가 정지한 런'; }
  else { st.className='status done'; $('#statusTxt').textContent='완료'; st.title=''; }
  const nEp=man.episodes||row.episodes||1;
  const ec=$('#epchips');
  if(nEp>1){
    ec.classList.remove('hidden'); ec.innerHTML='';
    // 라벨 정직화: 같은 시드 반복이면 '시도', 서로 다른 시드(구형 다문제)면 '에피소드'.
    ec.appendChild(el('span','lbl', seedsInfo(man).label));
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
  // 시드(문제 세트 = seed_base): 같은 시드면 같은 문제 → 재현·재활용의 근거. 라벨 있는 칩.
  if(Array.isArray(man.seeds)&&man.seeds.length&&typeof man.seeds[0]==='number')
    $('#meta').appendChild(chip('시드',man.seeds[0]));
}
function chip(k,v){const c=el('span','chip');c.innerHTML=esc(k)+' <b>'+esc(v)+'</b>';return c;}

function renderBoard(){
  if(!S.run){return;}
  const gme=gameId(); setBoardHead(gme);
  if(MULTI.has(gme)){ renderBoardG(gme); return; }
  const board=$('#board');
  const models=Object.keys(S.run.models);
  document.getElementById('board-wrap').classList.toggle('dense',models.length>6);
  document.getElementById('app').classList.toggle('dense',models.length>6);
  const rows=models.map(m=>({m,v:viewOf(m,S.episode)}));
  rows.sort((a,b)=>{
    const ta=sortTier(a.v), tb=sortTier(b.v);
    if(ta!==tb) return ta-tb;   // 정답 > 진행 중 > 턴 소진 실패 > 대기(진행도보다 우선)
    const ra=a.v.bestRank, rb=b.v.bestRank;
    if(ra==null&&rb==null){ const s=(b.v.lastSim||0)-(a.v.lastSim||0); return s||tieCmp(a,b); }
    if(ra==null) return 1; if(rb==null) return -1;
    if(ra!==rb) return ra-rb;
    const s=(b.v.lastSim||0)-(a.v.lastSim||0);   // 진행도 동률 → 턴·시간·토큰·slug 꼬리
    return s||tieCmp(a,b);
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
  if(S.focus===model) lane.classList.add('sel');
  lane.onclick=()=>openDrawer(model);

  // 게임 준비 중(예비 manifest 단계): 오라클 로딩 표시.
  if(v.phase==='preparing') return preparingLaneG(model,idx);

  // 대기 중: 완료/중단 런의 무데이터 레인(진짜 미시작)만 얇은 뮤트 행으로 유지.
  // 라이브 런의 무데이터 레인은 아래 실행 레인과 동일 구조·높이로 렌더('첫 턴 진행 중…').
  if(v.phase==='waiting' && !isLive()){
    lane.classList.add('waiting');
    lane.appendChild(el('div','pos',String(idx+1)));
    const who=el('div','who');
    who.appendChild(el('span','keydot'));
    const names=el('div','names');
    const aline=el('div','aline');
    aline.appendChild(el('span','alias',aliasOf(model)));
    const eff=effortOf(model); if(eff) aline.appendChild(el('span','eff',eff));
    names.appendChild(aline);
    names.appendChild(el('div','mid',model));
    names.appendChild(el('span','phase waiting','대기 중'));
    who.appendChild(names);
    lane.appendChild(who);
    return lane;
  }

  // 큰 숫자 색은 가독성 위해 하한을 둔다(후미 모델도 읽히게).
  lane.style.setProperty('--hc',v.bestRank!=null?heatColor(Math.max(0.28,closeness(v.bestRank))):'var(--muted)');
  const leader=(idx===0)&&v.bestRank!=null&&!v.solved&&isLive();
  if(leader) lane.classList.add('leader');
  if(v.solved) lane.classList.add('solved');

  lane.appendChild(el('div','pos',String(idx+1)));

  const who=el('div','who');
  who.appendChild(el('span','keydot'));
  const names=el('div','names');
  const aline=el('div','aline');
  aline.appendChild(el('span','alias',aliasOf(model)));
  const eff=effortOf(model); if(eff) aline.appendChild(el('span','eff',eff));
  const rb=reusedBadge(model); if(rb) aline.appendChild(rb);
  names.appendChild(aline);
  names.appendChild(el('div','mid',model));   // slug 전체(model@effort)
  const ex=isExhausted(v);   // 완주·미해결 = 'N턴 소진'으로 명시(중단과 구분)
  const phaseLabel = v.solved?'정답' : v.phase==='running'?'진행 중' : v.phase==='aborted'?'중단됨' : v.phase==='waiting'?'첫 턴 진행 중…' : ex?((v.turnsUsed||0)+'턴 소진'):'완료';
  const ph=el('span','phase '+(v.solved?'solvedp':ex?'exhausted':v.phase), phaseLabel);
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
  // 정답 레인: 순위 근거(정렬과 동일한 turnsUsed) 표기.
  // 진행 지표(제자리/갱신)는 라이브 & 실제 진행 중 레인에서만(정적·소진 레인엔 부적절).
  let su=null;
  if(v.solved) su=solveBasis(v);
  else if(isLive() && v.phase==='running') su=statusLine(v);
  rc.appendChild(su || el('div','stat-upd'));   // 배지 없어도 빈 슬롯으로 자리 유지(높이 불변)
  lane.appendChild(rc);

  // mid-col 슬롯을 항상 같은 개수로 배치(빈 슬롯도 min-height로 자리 유지 → 폴링 출렁임 제거).
  // 스트림 슬롯은 라이브 보드의 모든 활성 레인에 예약(진행/완료 무관 동일 높이).
  const mid=el('div','mid-col');
  if(isLive()) mid.appendChild(genLine(model));
  mid.appendChild(tickerEl(model,v));
  mid.appendChild(histStrip(model,v));
  mid.appendChild(laneCost(v));
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
  const strip=el('div','histstrip');
  if(!v.turns.length) return strip;   // 첫 턴 전에도 빈 스트립으로 자리 유지(높이 불변)
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
  const _th=$('.trend-head .tt'); if(_th)_th.textContent='접근 궤적';
  const _ax=$('.trend-head .ax'); if(_ax)_ax.textContent='턴별 근접도 · 위=정답(1위) · 참가자 색';
  if(legend) legend.innerHTML='';
  if(!host) return;
  host.innerHTML='';
  rows=(rows||[]).filter(r=>r.v.phase!=='waiting'&&r.v.phase!=='preparing');   // 대기·준비 중은 궤적/범례에서 제외(궤적 없음)
  if(!rows.length) return;
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

/* ==================== 멀티게임 디스패치 ==================== */
const MULTI=new Set(['ko-maze','ko-rulelab','ko-minefield']);
function gameId(){ return (S.run&&S.run.manifest&&S.run.manifest.game)||'ko-semantle'; }
function gmeta(g){ return (CFG.gameMeta&&CFG.gameMeta[g])||{kr:g,desc:'',max_turns:15}; }

function setBoardHead(g){
  const h=$('.board-head h2'); const lg=$('.board-head .legend'); if(!h||!lg) return;
  const man=(S.run&&S.run.manifest)||{};
  if(man.status==='preparing'){   // 예비 manifest 단계: 헤더·캡션에 준비 중임을 명시
    h.textContent='게임 준비 중 · 오라클 로딩(임베딩 모델·기준 어휘)';
    lg.innerHTML='<span class="k">준비가 끝나면 자동으로 시작됩니다</span>';
    return;
  }
  // 상태별 선두어(정적 런 부정직 방지): 라이브=실시간 / 완료=최종 / 중단=중단 시점.
  const aborted=(man.status==='failed'||!!man.failure);
  const statusWord = isLive() ? '실시간' : aborted ? '중단 시점' : '최종';
  // usage(시간·토큰) 유무로 정렬 사슬을 정직화 — 없는 런은 시간·토큰 순을 약속하지 않는다.
  const hasUsage=runHasUsage();
  const chain = hasUsage ? '턴 → 시간 → 토큰 순' : '턴 순 · 시간·토큰 기록 없는 런';
  const tie   = hasUsage ? '동률: 턴 → 시간 → 토큰' : '동률: 턴 · 시간·토큰 기록 없는 런';
  if(g==='ko-maze'){
    h.textContent=statusWord+' 탐험 · 목표 근접 → '+chain;
    lg.innerHTML='<span class="k">거리 ↓ = 목표 근접 · 탐사율</span><span class="k">'+tie+'</span>';
  }else if(g==='ko-rulelab'){
    h.textContent=statusWord+' 실험 · 규칙 규명 → '+chain;
    lg.innerHTML='<span class="k">정답 수 = 5문항 예측 적중 · 실험 = 관측</span><span class="k">'+tie+'</span>';
  }else if(g==='ko-minefield'){
    h.textContent=statusWord+' 순위 · 정답 근접 → '+chain;
    lg.innerHTML='<span class="k"><span class="swatch"></span><span class="arrow">순위 색 = 정답 근접</span></span><span class="k">폭사 최하 · '+tie+'</span>';
  }else{
    h.textContent=statusWord+' 순위 · 정답 근접 → '+chain;
    lg.innerHTML='<span class="k"><span class="swatch"></span><span class="arrow">순위 색 = 정답 근접</span></span><span class="k">1위 = 정답 · '+tie+'</span>';
  }
}

/* ---- 공통 base 뷰(에피소드 계산: viewOf의 phase 로직 공유) ---- */
function baseView(model,ep){
  const slot=S.ev[model]; const evs=slot?slot.events:[];
  const turns=[]; let end=null;
  for(const e of evs){ if(e.episode!==ep) continue; if(e.type==='episode_end') end=e; else if(e.type==='turn') turns.push(e); }
  const valid=turns.filter(t=>t.valid);
  const last=turns.length?turns[turns.length-1]:null;
  const live=(S.run.models[model]&&S.run.models[model].live)||null;
  const isCur=(ep===currentEpisode());
  const man=S.run.manifest||{};
  const runOver=(man.status==='done'||man.status==='failed'||!!man.finished_at);
  let phase='done';
  if(end){ phase=end.solved?'solved':'done'; }
  else if(live){ if(live.error||live.phase==='failed')phase='aborted'; else if(live.phase==='running')phase=runOver?'aborted':'running'; else phase='done'; }
  else if(turns.length===0)phase='waiting';
  else phase='done';
  if(man.status==='preparing' && !end && turns.length===0)phase='preparing';   // 예비 manifest 단계
  const cost=usageOf(turns);
  return {episode:ep,turns,valid,last,end,live,phase,isCur,
    solved:!!(end&&end.solved),
    maxTurns:(live&&live.max_turns)||man.max_turns||null,
    invalidCount:turns.filter(t=>!t.valid).length,
    turnsUsed:last?last.turn:(end?end.turns:0),
    durMs:cost.durMs, outTok:cost.outTok, costUsd:cost.costUsd};
}

function mazeView(model,ep){
  const b=baseView(model,ep);
  let dist=null,explored=null,bumps=null;
  const lastDist=[...b.valid].reverse().find(t=>t.dist!=null); if(lastDist)dist=lastDist.dist;
  const lastAny=b.turns.length?b.turns[b.turns.length-1]:null;
  if(lastAny){ if(lastAny.explored!=null)explored=lastAny.explored; if(lastAny.bumps!=null)bumps=lastAny.bumps; if(lastAny.dist!=null)dist=lastAny.dist; }
  if(b.live&&b.isCur){ if(b.live.dist!=null)dist=b.live.dist; if(b.live.explored!=null)explored=b.live.explored; if(b.live.bumps!=null)bumps=b.live.bumps; }
  if(b.end){ if(b.end.explored_ratio!=null)explored=b.end.explored_ratio; if(b.end.bumps!=null)bumps=b.end.bumps; dist=b.solved?0:(b.end.min_dist!=null?b.end.min_dist:dist); }
  const cells={}; let curPos=null;
  for(const t of b.valid){ if(!Array.isArray(t.pos))continue; const k=t.pos.join(','); const c=cells[k]||(cells[k]={x:t.pos[0],y:t.pos[1],open:new Set()}); String(t.open||'').split('·').filter(Boolean).forEach(d=>c.open.add(d)); curPos=t.pos; }
  return Object.assign(b,{game:'ko-maze',dist,explored,bumps,cells,curPos,
    target:b.end?b.end.target:null,
    distCurve:b.valid.filter(t=>t.dist!=null).map(t=>({turn:t.turn,dist:t.dist}))});
}

function rulelabView(model,ep){
  const b=baseView(model,ep);
  const tests=b.valid.filter(t=>t.kind==='test');
  const ansTurn=[...b.valid].reverse().find(t=>t.kind==='answer');
  let experiments=tests.length, answered=!!ansTurn, correct=ansTurn?ansTurn.correct:null;
  if(b.live&&b.isCur){ if(b.live.experiments!=null)experiments=b.live.experiments; if(b.live.answered)answered=true; }
  if(b.end){ if(b.end.experiments!=null)experiments=b.end.experiments; if(b.end.correct!=null)correct=b.end.correct; }
  const seen=new Set(); let dup=0;
  for(const t of tests){ const k=(t.input||[]).join(','); if(seen.has(k))dup++; else seen.add(k); }
  if(b.end&&b.end.duplicate_tests!=null)dup=b.end.duplicate_tests;
  return Object.assign(b,{game:'ko-rulelab',experiments,answered,correct,dup,
    rule:b.end?b.end.target:null, answer:ansTurn?ansTurn.answer:null,
    tests:tests.map(t=>({input:t.input,output:t.output})), finished:!!b.end});
}

function minefieldView(model,ep){
  const b=baseView(model,ep);
  let bestRank=null;
  for(const t of b.valid){ if(t.rank!=null&&(bestRank==null||t.rank<bestRank))bestRank=t.rank; }
  if(b.end&&b.end.best_rank!=null)bestRank=(bestRank==null)?b.end.best_rank:Math.min(bestRank,b.end.best_rank);
  let lives=null;
  const lastLife=[...b.valid].reverse().find(t=>t.lives!=null); if(lastLife)lives=lastLife.lives;
  if(b.live&&b.isCur&&b.live.lives!=null)lives=b.live.lives;
  if(b.end&&b.end.lives_left!=null)lives=b.end.lives_left;
  const booms=b.end?b.end.booms:b.valid.filter(t=>t.mine_event==='boom').length;
  const warns=b.end?b.end.warns:b.valid.filter(t=>t.mine_event==='warn').length;
  const lastValidSim=[...b.valid].reverse().find(t=>t.similarity!=null);
  const lastSim=(b.live&&b.isCur&&b.live.last_similarity!=null)?b.live.last_similarity:(lastValidSim?lastValidSim.similarity:null);
  const mined=!!(b.end&&!b.solved&&lives===0);
  const maxLives=(S.run.manifest&&S.run.manifest.oracle&&S.run.manifest.oracle.lives)||3;
  return Object.assign(b,{game:'ko-minefield',bestRank,lives,booms,warns,lastSim,mined,maxLives,
    target:b.end?b.end.target:null, mines:b.end?b.end.mines:null});
}

/* ---- 공통 정렬 꼬리 · 자원 표기 ---- */
function _ascMissLast(x,y){   // 오름차순, null(결측)은 이 단계에서 최하
  const xn=(x==null),yn=(y==null);
  if(xn&&yn) return 0; if(xn) return 1; if(yn) return -1; return x-y;
}
// 게임별 진행도 비교 뒤 붙는 공통 꼬리: ① 턴↑ ② 시간↑ ③ 출력 토큰↑ ④ slug↑.
// 결측(usage 없는 구형 런)은 해당 단계에서 최하로 두되 그 위 단계 순서는 유지.
function tieCmp(a,b){
  const av=a.v, bv=b.v;
  const ta=av.turnsUsed||0, tb=bv.turnsUsed||0; if(ta!==tb) return ta-tb;
  const d=_ascMissLast(av.durMs, bv.durMs); if(d) return d;
  const o=_ascMissLast(av.outTok, bv.outTok); if(o) return o;
  return a.m<b.m?-1:a.m>b.m?1:0;
}
function fmtDur(ms){                    // m:ss (1시간 이상 h:mm:ss)
  if(ms==null) return null;
  const s=Math.max(0,Math.round(ms/1000));
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=s%60;
  const p2=n=>String(n).padStart(2,'0');
  return h>0 ? h+':'+p2(m)+':'+p2(ss) : m+':'+p2(ss);
}
function fmtTok(n){                      // 1000 이상 k 표기
  if(n==null) return null;
  if(n>=1000) return (n/1000).toFixed(n>=10000?0:1)+'k';
  return String(n);
}
function fmtCost(usd){                   // 소액도 유효자리 살림. 0이면 $0(무과금/로컬 정직). 결측 → null(칩 생략)
  if(typeof usd!=='number'||!isFinite(usd)) return null;
  if(usd===0) return '$0';
  if(usd>=1) return '$'+usd.toFixed(2);
  if(usd>=0.1) return '$'+usd.toFixed(3);
  return '$'+usd.toFixed(4);
}
// 레인 자원 지표(정보량 비례 — 작게, 라벨 포함). 결측 단계는 칩 생략하되 슬롯(빈 costline)은 유지.
function laneCost(v){
  const d=fmtDur(v.durMs), t=fmtTok(v.outTok), cu=fmtCost(v.costUsd);
  const w=el('div','costline');   // 데이터 없어도 빈 슬롯 반환(높이 예약 → 폴링 시 출렁임 방지)
  if(d!=null){ const c=el('span','cchip'); c.innerHTML='시간 <b>'+esc(d)+'</b>'; w.appendChild(c); }
  if(t!=null){ const c=el('span','cchip'); c.innerHTML='출력 <b>'+esc(t)+'</b> tok'; w.appendChild(c); }
  if(cu!=null){ const c=el('span','cchip cost'); c.innerHTML='<b>'+esc(cu)+'</b>'; w.appendChild(c); }   // 결측 usage → 칩 생략(0으로 지어내지 않음)
  return w;
}

/* ---- 정렬 비교자 ---- */
function cmpFor(g){
  // 티어 우선: 정답 > 진행 중 > 턴 소진 실패 > 대기. 같은 티어 안에서만 게임별 진행도 사슬.
  const tier=(a,b)=>{ const ta=sortTier(a.v),tb=sortTier(b.v); return ta!==tb?ta-tb:0; };
  if(g==='ko-maze') return (a,b)=>{ const t=tier(a,b); if(t)return t;
    if(a.v.solved!==b.v.solved)return a.v.solved?-1:1;
    const ad=a.v.dist,bd=b.v.dist;
    if(ad==null&&bd==null){ const e=(b.v.explored||0)-(a.v.explored||0); return e||tieCmp(a,b); }
    if(ad==null)return 1; if(bd==null)return -1;
    if(ad!==bd)return ad-bd;
    const e=(b.v.explored||0)-(a.v.explored||0); return e||tieCmp(a,b); };
  if(g==='ko-rulelab') return (a,b)=>{ const t=tier(a,b); if(t)return t;
    const ac=a.v.answered&&a.v.correct!=null?a.v.correct:-1;
    const bc=b.v.answered&&b.v.correct!=null?b.v.correct:-1;
    if(ac!==bc)return bc-ac;
    const e=(b.v.experiments||0)-(a.v.experiments||0); return e||tieCmp(a,b); };
  return (a,b)=>{ const t=tier(a,b); if(t)return t;
    if(a.v.mined!==b.v.mined)return a.v.mined?1:-1;   // 폭사 최하(같은 실패 티어 안에서)
    const ra=a.v.bestRank,rb=b.v.bestRank;
    if(ra==null&&rb==null){ const s=(b.v.lastSim||0)-(a.v.lastSim||0); return s||tieCmp(a,b); }
    if(ra==null)return 1; if(rb==null)return -1;
    if(ra!==rb)return ra-rb;
    const s=(b.v.lastSim||0)-(a.v.lastSim||0); return s||tieCmp(a,b); };
}

/* ---- 공통 레인 조각 ---- */
function laneWho(model){
  const who=el('div','who'); who.appendChild(el('span','keydot'));
  const names=el('div','names');
  const aline=el('div','aline');
  aline.appendChild(el('span','alias',aliasOf(model)));
  const eff=effortOf(model); if(eff) aline.appendChild(el('span','eff',eff));
  const rb=reusedBadge(model); if(rb) aline.appendChild(rb);
  names.appendChild(aline);
  names.appendChild(el('div','mid',model));
  who.appendChild(names);
  return {who,names};
}
function laneTurns(v){
  const turns=el('div','turns');
  const tn=el('div','tn'); const nt=v.last?v.last.turn:(v.end?v.end.turns:0);
  tn.innerHTML='<span>'+esc(nt)+'</span><span class="sep">/'+esc(v.maxTurns||'?')+'</span>';
  turns.appendChild(tn); turns.appendChild(el('div','lab','턴'));
  if(v.invalidCount>0) turns.appendChild(el('div','inv','무효 '+v.invalidCount));
  return turns;
}
function waitingLaneG(model,idx){
  const lane=el('div','lane waiting'); lane.dataset.slug=model;
  lane.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  lane.onclick=()=>openDrawer(model);
  lane.appendChild(el('div','pos',String(idx+1)));
  const {who,names}=laneWho(model);
  names.appendChild(el('span','phase waiting','대기 중'));
  lane.appendChild(who);
  return lane;
}
// 게임 준비 중(오라클 로딩): 순위/지표 없이 정체성만, 준비 중임을 명시(대기와 구분).
// 모든 게임 보드가 공유(semantle laneEl + 멀티게임 레인 렌더러).
function preparingLaneG(model,idx){
  const lane=el('div','lane preparing'); lane.dataset.slug=model;
  lane.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  lane.onclick=()=>openDrawer(model);
  lane.appendChild(el('div','pos','–'));   // 순위 없음(준비 중)
  const {who,names}=laneWho(model);
  names.appendChild(el('span','phase preparing','게임 준비 중(오라클 로딩)'));
  lane.appendChild(who);
  return lane;
}
function gmChip(k,val,warn){ const c=el('span','gm-chip'+(warn?' warnv':'')); c.innerHTML=esc(k)+' <b>'+esc(val)+'</b>'; return c; }

/* ---- 보드 디스패치 ---- */
function renderBoardG(g){
  const board=$('#board'); if(!S.run)return;
  const models=Object.keys(S.run.models);
  document.getElementById('board-wrap').classList.toggle('dense',models.length>6);
  document.getElementById('app').classList.toggle('dense',models.length>6);
  const vf=g==='ko-maze'?mazeView:g==='ko-rulelab'?rulelabView:minefieldView;
  const lf=g==='ko-maze'?laneMaze:g==='ko-rulelab'?laneRulelab:laneMine;
  const rows=models.map(m=>({m,v:vf(m,S.episode)}));
  rows.sort(cmpFor(g));
  board.innerHTML='';
  rows.forEach((row,idx)=>board.appendChild(lf(row.m,row.v,idx)));
  (g==='ko-maze'?renderTrendMaze:g==='ko-rulelab'?renderTrendRulelab:renderTrend)(rows);
  paintStreams();
}

/* ---- ko-maze: 레인 · 미니맵 · 트렌드 ---- */
function laneMaze(model,v,idx){
  if(v.phase==='preparing') return preparingLaneG(model,idx);
  if(v.phase==='waiting' && !isLive()) return waitingLaneG(model,idx);   // 라이브면 아래 전체 구조로
  const lane=el('div','lane maze'); lane.dataset.slug=model;
  lane.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  if(S.focus===model)lane.classList.add('sel');
  if(v.solved)lane.classList.add('solved');
  if((idx===0)&&!v.solved&&v.dist!=null&&isLive())lane.classList.add('leader');
  lane.onclick=()=>openDrawer(model);
  lane.appendChild(el('div','pos',String(idx+1)));
  const {who,names}=laneWho(model);
  const ex=isExhausted(v);
  const label=v.solved?'도착':v.phase==='running'?'탐험 중':v.phase==='aborted'?'중단됨':v.phase==='waiting'?'첫 턴 진행 중…':ex?((v.turnsUsed||0)+'턴 소진'):'완료';
  names.appendChild(el('span','phase '+(v.solved?'solvedp':ex?'exhausted':v.phase),label));
  lane.appendChild(who);
  const gm=el('div','gmetrics');
  gm.appendChild(el('div','gm-cap','남은 거리'));
  const big=el('div','gm-big'+(v.dist==null?' na':'')+(v.solved?' goal':''));
  if(v.dist!=null){ big.appendChild(document.createTextNode(String(v.dist))); big.appendChild(el('span','gm-unit',v.solved?'· 도착':'칸')); }
  else big.textContent='—';
  gm.appendChild(big);
  const sub=el('div','gm-sub');
  sub.appendChild(gmChip('탐사',v.explored!=null?(v.explored*100).toFixed(0)+'%':'—'));
  sub.appendChild(gmChip('충돌',v.bumps!=null?String(v.bumps):'0',v.bumps>0));
  gm.appendChild(sub); lane.appendChild(gm);
  const midc=el('div','mid-col');
  if(isLive()) midc.appendChild(genLine(model));   // 라이브 보드: 스트림 슬롯 항상 예약
  const mw=el('div','mm-wrap');
  const mmcol=el('div','mm-col');
  mmcol.appendChild(mazeMiniMap(v,46));
  mmcol.appendChild(el('div','mm-cap','7×7 지도'));
  mw.appendChild(mmcol);
  mw.appendChild(mazeTick(v));
  midc.appendChild(mw);
  midc.appendChild(laneCost(v));                    // 빈 슬롯도 유지(자리 예약)
  lane.appendChild(midc);
  lane.appendChild(laneTurns(v));
  return lane;
}
function mazeTick(v){
  const t=el('div','maze-tick'); const last=v.last;
  if(!last){ t.innerHTML='<span class="verb">이동 대기</span>'; return t; }
  if(!last.valid){ t.innerHTML='<span class="verb">최근</span><span style="color:var(--crit);font-weight:700">형식 오류</span>'; return t; }
  const pos=(last.pos||[]).join(',');
  if(last.ok===false) t.innerHTML='<span class="verb">최근</span><b>'+esc(last.move)+'</b> → 벽에 막힘 <span class="brg">('+esc(pos)+')</span>';
  else t.innerHTML='<span class="verb">최근</span><b>'+esc(last.move)+'</b> → ('+esc(pos)+') <span class="brg">방위 '+esc(last.bearing||'')+'</span>';
  return t;
}
function mazeMiniMap(v,px){
  const NS='http://www.w3.org/2000/svg'; const N=7,U=10,PAD=1.2;
  const mk=(tag,at,cls)=>{const e=document.createElementNS(NS,tag);if(cls)e.setAttribute('class',cls);for(const k in at)e.setAttribute(k,at[k]);return e;};
  const svg=mk('svg',{viewBox:`0 0 ${N*U} ${N*U}`,width:px,height:px},'minimap');
  const ttl=mk('title',{}); ttl.textContent='탐험 지도 7×7 (x=동쪽, y=남쪽)'; svg.appendChild(ttl);
  svg.appendChild(mk('rect',{x:0,y:0,width:N*U,height:N*U},'mm-bg'));
  const cells=v.cells||{}; const cur=v.curPos?v.curPos.join(','):null;
  Object.values(cells).forEach(c=>{
    const cx=c.x*U,cy=c.y*U,mid=U/2;
    svg.appendChild(mk('rect',{x:cx+PAD,y:cy+PAD,width:U-2*PAD,height:U-2*PAD,rx:1.5},'mm-cell'));
    c.open.forEach(d=>{ let x2=cx+mid,y2=cy+mid;
      if(d==='북')y2=cy; else if(d==='남')y2=cy+U; else if(d==='동')x2=cx+U; else if(d==='서')x2=cx;
      svg.appendChild(mk('line',{x1:cx+mid,y1:cy+mid,x2,y2},'mm-open')); });
  });
  if(v.target){ const p=String(v.target).split(',').map(Number); if(p.length===2&&!isNaN(p[0])) svg.appendChild(mk('rect',{x:p[0]*U+PAD,y:p[1]*U+PAD,width:U-2*PAD,height:U-2*PAD,rx:1.5},'mm-goal')); }
  if(cur){ const p=cur.split(',').map(Number); const dot=mk('circle',{cx:p[0]*U+U/2,cy:p[1]*U+U/2,r:2.6},'mm-cur'); dot.style.fill='var(--mc)'; svg.appendChild(dot); }
  return svg;
}
function renderTrendMaze(rows){
  const host=$('#trendBody'),legend=$('#trendLegend');
  const th=$('.trend-head .tt'); if(th)th.textContent='탐험 궤적';
  const ax=$('.trend-head .ax'); if(ax)ax.textContent='턴별 남은 거리 · 아래=목표 도달 · 참가자 색';
  if(legend)legend.innerHTML=''; if(!host)return; host.innerHTML='';
  rows=(rows||[]).filter(r=>r.v.phase!=='waiting'&&r.v.phase!=='preparing'); if(!rows.length)return;
  const NS='http://www.w3.org/2000/svg';
  const mk=(tag,at,cls)=>{const e=document.createElementNS(NS,tag);if(cls)e.setAttribute('class',cls);for(const k in at)e.setAttribute(k,at[k]);return e;};
  const series=rows.map(r=>({color:`var(${S.colorOf[r.m]})`,alias:slugLabel(r.m),solved:!!r.v.solved,pts:(r.v.distCurve||[])}));
  if(legend)series.forEach(s=>{const k=el('span','k');const sw=el('span','swatch');sw.style.background=s.color;k.appendChild(sw);k.appendChild(el('span','nm',s.alias));legend.appendChild(k);});
  const W=Math.max(220,Math.round(host.clientWidth||640)),H=Math.max(120,Math.round(host.clientHeight||160));
  const svg=mk('svg',{viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'none'},'trend-svg');
  if(!series.some(s=>s.pts.length>0)){ const tx=mk('text',{x:W/2,y:H/2,'text-anchor':'middle','dominant-baseline':'middle'},'empty'); tx.textContent='이동 대기'; svg.appendChild(tx); host.appendChild(svg); return; }
  const mL=44,mR=14,mT=24,mB=30; const plotW=Math.max(10,W-mL-mR),plotH=Math.max(10,H-mT-mB);
  let xmax=2,dmax=1; series.forEach(s=>s.pts.forEach(p=>{if(p.turn>xmax)xmax=p.turn; if(p.dist>dmax)dmax=p.dist;}));
  const X=t=>mL+((t-1)/(xmax-1))*plotW; const Y=d=>mT+(1-d/dmax)*plotH;
  svg.appendChild(mk('line',{x1:mL,y1:Y(0),x2:mL+plotW,y2:Y(0)},'grid'));
  svg.appendChild(mk('line',{x1:mL,y1:Y(dmax),x2:mL+plotW,y2:Y(dmax)},'axis'));
  const l0=mk('text',{x:mL-7,y:Y(0)+3,'text-anchor':'end'},'lbl'); l0.textContent='목표'; svg.appendChild(l0);
  const ld=mk('text',{x:mL-7,y:Y(dmax)+3,'text-anchor':'end'},'lbl'); ld.textContent=String(dmax); svg.appendChild(ld);
  const yt=mk('text',{x:mL,y:12,'text-anchor':'start'},'axtitle'); yt.textContent='남은 거리 (아래=목표)'; svg.appendChild(yt);
  const step=Math.max(1,Math.ceil((xmax-1)/5));
  for(let t=1;t<=xmax;t+=step){ const tk=mk('text',{x:X(t),y:Y(0)+15,'text-anchor':'middle'},'lbl'); tk.textContent=String(t); svg.appendChild(tk); }
  const xt=mk('text',{x:mL+plotW/2,y:H-4,'text-anchor':'middle'},'axtitle'); xt.textContent='턴 →'; svg.appendChild(xt);
  series.forEach(s=>{ if(!s.pts.length)return;
    if(s.pts.length>=2){ const pl=mk('polyline',{points:s.pts.map(p=>X(p.turn).toFixed(1)+','+Y(p.dist).toFixed(1)).join(' ')},'line'); pl.style.stroke=s.color; svg.appendChild(pl); }
    const last=s.pts[s.pts.length-1];
    const dot=mk('circle',{cx:X(last.turn).toFixed(1),cy:Y(last.dist).toFixed(1),r:s.solved?5:3.3},'dot'); dot.style.fill=s.color;
    const tt=mk('title',{}); tt.textContent=`${s.alias} · ${last.turn}턴 · 거리 ${last.dist}`; dot.appendChild(tt); svg.appendChild(dot); });
  host.appendChild(svg);
}

/* ---- ko-rulelab: 레인 · 실험 로그 · 트렌드 ---- */
function laneRulelab(model,v,idx){
  if(v.phase==='preparing') return preparingLaneG(model,idx);
  if(v.phase==='waiting' && !isLive()) return waitingLaneG(model,idx);   // 라이브면 아래 전체 구조로
  const lane=el('div','lane rulelab'); lane.dataset.slug=model;
  lane.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  if(S.focus===model)lane.classList.add('sel');
  if(v.solved)lane.classList.add('solved');
  lane.onclick=()=>openDrawer(model);
  lane.appendChild(el('div','pos',String(idx+1)));
  const {who,names}=laneWho(model);
  const ex=isExhausted(v)&&!v.answered;   // 답변했으면 '답변 완료', 미답변으로 소진하면 'N턴 소진'
  const label=v.solved?'규명':v.answered?'답변 완료':v.phase==='running'?'실험 중':v.phase==='aborted'?'중단됨':v.phase==='waiting'?'첫 턴 진행 중…':ex?((v.turnsUsed||0)+'턴 소진'):'완료';
  names.appendChild(el('span','phase '+(v.solved?'solvedp':ex?'exhausted':v.phase),label));
  lane.appendChild(who);
  const gm=el('div','gmetrics');
  if(v.answered&&v.correct!=null){
    gm.appendChild(el('div','gm-cap','정답 수'));
    const big=el('div','gm-big'+(v.correct===5?' goal':'')); big.appendChild(document.createTextNode(String(v.correct))); big.appendChild(el('span','gm-unit','/5')); gm.appendChild(big);
    const sub=el('div','gm-sub'); sub.appendChild(gmChip('실험',String(v.experiments))); if(v.dup>0)sub.appendChild(gmChip('중복',String(v.dup),true)); gm.appendChild(sub);
  }else{
    gm.appendChild(el('div','gm-cap','실험 수'));
    const big=el('div','gm-big'); big.textContent=String(v.experiments); gm.appendChild(big);
    const sub=el('div','gm-sub'); sub.appendChild(gmChip('상태',v.finished?'미답변':'실험 중',v.finished)); gm.appendChild(sub);
  }
  lane.appendChild(gm);
  const midc=el('div','mid-col');
  if(isLive()) midc.appendChild(genLine(model));    // 라이브 보드: 스트림 슬롯 항상 예약
  midc.appendChild(expLog(v));
  midc.appendChild(laneCost(v));                     // 빈 슬롯도 유지(자리 예약)
  lane.appendChild(midc);
  lane.appendChild(laneTurns(v));
  return lane;
}
function expLog(v){
  const wrap=el('div','explog');
  const recent=(v.tests||[]).slice(-5);
  if(!recent.length&&!v.answered){ wrap.appendChild(el('span','verb','실험 대기')); return wrap; }
  recent.forEach(t=>{ const c=el('span','exp-chip'); c.innerHTML='<b>'+esc((t.input||[]).join(', '))+'</b> → <span class="out">'+esc(t.output)+'</span>'; wrap.appendChild(c); });
  if(v.answered){ const a=el('span','exp-ans'+(v.correct===5?' ok':'')); a.textContent='제출 '+(v.correct!=null?v.correct+'/5':'—'); wrap.appendChild(a); }
  return wrap;
}
function renderTrendRulelab(rows){
  const host=$('#trendBody'),legend=$('#trendLegend');
  const th=$('.trend-head .tt'); if(th)th.textContent='실험 · 정답 비교';
  const ax=$('.trend-head .ax'); if(ax)ax.textContent='참가자별 실험 횟수(막대) · 정답 예측 수(/5)';
  if(legend)legend.innerHTML=''; if(!host)return; host.innerHTML='';
  rows=(rows||[]).filter(r=>r.v.phase!=='waiting'&&r.v.phase!=='preparing'); if(!rows.length)return;
  const maxExp=Math.max(1,...rows.map(r=>r.v.experiments||0));
  const box=el('div','rl-bars');
  rows.forEach(r=>{
    const row=el('div','rl-row');
    const nm=el('div','rl-nm'); const sw=el('span','rl-sw'); sw.style.background=`var(${S.colorOf[r.m]})`; nm.appendChild(sw); nm.appendChild(el('span','rl-nmt',slugLabel(r.m))); row.appendChild(nm);
    const bw=el('div','rl-barwrap');
    const bar=el('div','rl-bar'); bar.style.width=((r.v.experiments||0)/maxExp*100).toFixed(1)+'%'; bar.style.background=`var(${S.colorOf[r.m]})`; bw.appendChild(bar);
    bw.appendChild(el('span','rl-bl',(r.v.experiments||0)+' 실험'+(r.v.dup?(' · 중복 '+r.v.dup):''))); row.appendChild(bw);
    const cr=el('div','rl-correct');
    if(r.v.answered&&r.v.correct!=null){ for(let i=0;i<5;i++)cr.appendChild(el('span','rl-pip'+(i<r.v.correct?' on':''))); cr.appendChild(el('span','rl-cn',r.v.correct+'/5')); }
    else cr.appendChild(el('span','rl-cn muted',r.v.finished?'미답변':'실험 중'));
    row.appendChild(cr); box.appendChild(row);
  });
  host.appendChild(box);
}

/* ---- ko-minefield: 레인 · 목숨 · 이력 ---- */
function laneMine(model,v,idx){
  if(v.phase==='preparing') return preparingLaneG(model,idx);
  if(v.phase==='waiting' && !isLive()) return waitingLaneG(model,idx);   // 라이브면 아래 전체 구조로
  const lane=el('div','lane mine'); lane.dataset.slug=model;
  lane.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  lane.style.setProperty('--hc', v.bestRank!=null?heatColor(Math.max(0.28,closeness(v.bestRank))):'var(--muted)');
  if(S.focus===model)lane.classList.add('sel');
  if(v.solved)lane.classList.add('solved');
  if(v.mined)lane.classList.add('mined');
  if((idx===0)&&!v.solved&&!v.mined&&v.bestRank!=null&&isLive())lane.classList.add('leader');
  lane.onclick=()=>openDrawer(model);
  lane.appendChild(el('div','pos',String(idx+1)));
  const {who,names}=laneWho(model);
  const ex=isExhausted(v)&&!v.mined;   // 폭사는 '폭사', 그 외 완주·미해결은 'N턴 소진'
  const label=v.solved?'정답':v.mined?'폭사':v.phase==='running'?'진행 중':v.phase==='aborted'?'중단됨':v.phase==='waiting'?'첫 턴 진행 중…':ex?((v.turnsUsed||0)+'턴 소진'):'완료';
  names.appendChild(el('span','phase '+(v.solved?'solvedp':v.mined?'aborted':ex?'exhausted':v.phase),label));
  lane.appendChild(who);
  const rc=el('div','rankcell');
  rc.appendChild(el('div','cap','최고 순위'));
  const big=el('div','big'+(v.bestRank==null?' na':''));
  if(v.bestRank!=null){ big.appendChild(document.createTextNode(String(v.bestRank))); big.appendChild(el('span','unit','위')); big.appendChild(el('span','denom','/ '+S.refWords)); }
  else big.textContent='—';
  rc.appendChild(big);
  let su=null;
  if(v.solved) su=solveBasis(v);          // 정답 레인: 순위 근거(turnsUsed) 표기
  else if(isLive() && v.phase==='running') su=statusLine(v);   // 진행 지표는 진행 중 레인만
  rc.appendChild(su || el('div','stat-upd'));   // 배지 없어도 빈 슬롯으로 자리 유지
  lane.appendChild(rc);
  const midc=el('div','mid-col');
  if(isLive()) midc.appendChild(genLine(model));   // 라이브 보드: 스트림 슬롯 항상 예약
  midc.appendChild(livesEl(v));
  midc.appendChild(mineTicker(model,v));
  midc.appendChild(mineHist(v));                    // 빈 이력이면 빈 스트립(자리 예약)
  midc.appendChild(laneCost(v));                    // 빈 슬롯도 유지(자리 예약)
  lane.appendChild(midc);
  lane.appendChild(laneTurns(v));
  return lane;
}
function livesEl(v){
  const w=el('div','lives'); w.appendChild(el('span','lv-lab','목숨'));
  const n=v.maxLives||3; const cur=(v.lives==null)?n:v.lives;
  for(let i=0;i<n;i++) w.appendChild(el('span','pip '+(i<cur?'on':'off')));
  if(v.mined) w.appendChild(el('span','lv-out','폭사'));
  return w;
}
function mineTicker(model,v){
  const t=el('div','ticker');
  if(v.solved&&v.target){ t.appendChild(el('span','verb','정답')); const tg=el('span','target'); tg.innerHTML='정답 <b>'+esc(v.target)+'</b>'; t.appendChild(tg); return t; }
  const last=v.last;
  if(!last){ const w=el('span','word','추측 대기'); w.style.color='var(--muted)'; t.appendChild(w); return t; }
  if(!last.valid){ t.appendChild(el('span','verb','최근'));
    if(last.guess){ t.appendChild(el('span','word',last.guess)); t.appendChild(el('span','bad dup','중복')); }
    else { const w=el('span','word',((last.raw||'').replace(/\n/g,' ').trim().slice(0,18))||'—'); w.style.color='var(--muted)'; t.appendChild(w); t.appendChild(el('span','bad fmt','형식 오류')); }
    return t; }
  t.appendChild(el('span','verb','최근'));
  t.appendChild(el('span','word',last.guess));
  if(last.mine_event==='boom'){ t.appendChild(el('span','bad boom','지뢰 폭발')); return t; }
  if(last.rank!=null) t.appendChild(el('span','rk','· '+last.rank+'위'));
  if(last.similarity!=null){ const sim=el('span','sim'); sim.innerHTML='· 유사도 <b>'+(last.similarity*100).toFixed(1)+'</b>%'; t.appendChild(sim); }
  if(last.mine_event==='warn') t.appendChild(el('span','bad warn','지뢰 경보'));
  return t;
}
function mineHist(v){
  const strip=el('div','histstrip');
  if(!v.turns.length) return strip;   // 첫 턴 전에도 빈 스트립으로 자리 유지
  strip.appendChild(el('span','histlab','이력'));
  let shown=0;
  for(const t of [...v.turns].reverse()){
    if(shown>=8) break;
    if(t.valid){
      if(t.mine_event==='boom'){ const c=el('span','hchip boom','💥'); c.title='지뢰 폭발'; strip.appendChild(c); shown++; continue; }
      const cl=closeness(t.rank); const chip=el('span','hchip'+(t.mine_event==='warn'?' warn':'')); const rgb=heatRGB(cl); chip.style.background=`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`; chip.style.color=inkOn(rgb);
      chip.appendChild(el('span','hw',t.guess)); if(t.rank!=null)chip.appendChild(el('span','hr',t.rank+'위')); strip.appendChild(chip); shown++;
    } else { const chip=el('span','hchip inv',t.guess?'✕':'무효'); chip.title=t.guess?'중복 추측':'형식 오류'; strip.appendChild(chip); }
  }
  return strip;
}

/* ---- 게임별 드로어 ---- */
function renderDrawerG(g,model){
  const v=(g==='ko-maze'?mazeView:g==='ko-rulelab'?rulelabView:minefieldView)(model,S.episode);
  const dw=$('#drawer'); dw.innerHTML='';
  dw.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  const head=el('div','dw-head');
  const top=el('div','top'); top.appendChild(el('span','keydot')); top.appendChild(el('span','alias',aliasOf(model)));
  const deff=effortOf(model); if(deff)top.appendChild(el('span','eff',deff));
  const drb=reusedBadge(model); if(drb)top.appendChild(drb);
  const cl=el('button','close','✕'); cl.onclick=closeDrawer; top.appendChild(cl); head.appendChild(top);
  head.appendChild(el('div','mid',model+' · '+gmeta(g).kr+' · 에피소드 '+S.episode));
  const stats=el('div','dw-stats');
  if(g==='ko-maze'){
    stats.appendChild(statEl(v.dist!=null?String(v.dist):'—','남은 거리'));
    stats.appendChild(statEl(v.explored!=null?(v.explored*100).toFixed(0)+'%':'—','탐사율'));
    stats.appendChild(statEl(v.bumps!=null?String(v.bumps):'0','벽 충돌'));
    stats.appendChild(statEl(String(v.valid.length),'이동'));
  }else if(g==='ko-rulelab'){
    stats.appendChild(statEl(String(v.experiments),'실험'));
    stats.appendChild(statEl(String(v.dup),'중복 실험'));
    stats.appendChild(statEl(v.answered&&v.correct!=null?v.correct+'/5':'—','정답 수'));
    stats.appendChild(statEl(v.answered?'제출':(v.finished?'미답변':'진행'),'답변'));
  }else{
    const brT=statEl(v.bestRank!=null?v.bestRank+'위 / '+S.refWords:'—','최고 순위');
    if(v.bestRank!=null)brT.querySelector('.v').style.color=heatColor(Math.max(0.28,closeness(v.bestRank)));
    stats.appendChild(brT);
    stats.appendChild(statEl(v.lives!=null?v.lives+' / '+(v.maxLives||3):'—','목숨'));
    stats.appendChild(statEl(String(v.booms||0),'지뢰 폭발'));
    stats.appendChild(statEl(String(v.warns||0),'경보'));
  }
  head.appendChild(stats);
  let revHtml=null;
  if(g==='ko-maze'&&v.target) revHtml='목표 좌표 <b>('+esc(v.target)+')</b> · '+(v.solved?'도착':'미도착');
  else if(g==='ko-rulelab'&&v.rule) revHtml='숨은 규칙 <b>'+esc(v.rule)+'</b> · 정답 '+(v.correct!=null?v.correct:0)+'/5';
  else if(g==='ko-minefield'&&v.target) revHtml='정답 <b>'+esc(v.target)+'</b>'+(v.mines?' · 지뢰 '+v.mines.map(esc).join(', '):'')+' · '+(v.solved?'해결':v.mined?'목숨 소진':'미해결');
  if(revHtml){ const rev=el('div','dw-reveal show'); rev.innerHTML=revHtml; head.appendChild(rev); }
  dw.appendChild(head);
  const gen=el('div','dw-gen');
  gen.appendChild(el('div','lbl','생성 중 (공개 출력)'));
  const gw=el('div','genfull-wrap'); gw.appendChild(el('span','genfull')); gw.appendChild(el('span','cursor')); gen.appendChild(gw);
  gen.appendChild(el('div','dw-gen-wait','아직 공개 출력이 없어요 (생성 중…)'));
  dw.appendChild(gen); paintDrawerStream(model);
  const stream=el('div','dw-stream');
  stream.appendChild(el('div','lbl','행동 기록 (공개 출력)'));
  const desc=[...v.turns].reverse();
  if(!desc.length) stream.appendChild(el('div','mid','아직 행동이 없습니다.'));
  const rf=g==='ko-maze'?mazeTurnRow:g==='ko-rulelab'?rulelabTurnRow:mineTurnRow;
  desc.forEach(t=>stream.appendChild(rf(t)));
  dw.appendChild(stream);
}
function mazeTurnRow(t){
  const row=el('div','turn-row'); if(!t.valid)row.classList.add('fmt');
  row.appendChild(el('div','t','#'+t.turn));
  const g=el('div','g');
  if(t.valid){ const pos=(t.pos||[]).join(',');
    if(t.ok===false){ g.appendChild(el('div','w',esc(t.move)+' → 벽에 막힘')); g.appendChild(el('div','e','('+esc(pos)+') 이동 실패')); }
    else { g.appendChild(el('div','w',esc(t.move)+' → ('+esc(pos)+')')); g.appendChild(el('div','e','열림 '+esc(t.open||'')+' · 방위 '+esc(t.bearing||''))); } }
  else { g.appendChild(el('div','w',((t.raw||'').replace(/\n/g,' ').trim().slice(0,22))||'(빈 응답)')); g.appendChild(el('div','e','형식 오류')); }
  row.appendChild(g);
  const r=el('div','r'); r.innerHTML=(t.valid&&t.dist!=null)?'<div class="rk">거리 '+t.dist+'</div>':'<div class="rk">—</div>'; row.appendChild(r);
  return row;
}
function rulelabTurnRow(t){
  const row=el('div','turn-row'); if(!t.valid)row.classList.add('fmt');
  row.appendChild(el('div','t','#'+t.turn));
  const g=el('div','g');
  if(t.valid&&t.kind==='test') g.appendChild(el('div','w','실험 '+(t.input||[]).join(', ')+' → '+esc(t.output)));
  else if(t.valid&&t.kind==='answer'){ g.appendChild(el('div','w','제출 '+(t.answer||[]).join(', '))); g.appendChild(el('div','e','5문항 예측 제출(1회 한정)')); }
  else { g.appendChild(el('div','w',((t.raw||'').replace(/\n/g,' ').trim().slice(0,22))||'(빈 응답)')); g.appendChild(el('div','e','형식 오류')); }
  row.appendChild(g);
  const r=el('div','r');
  if(t.valid&&t.kind==='answer')r.innerHTML='<div class="rk">'+(t.correct!=null?t.correct:0)+'/5</div>';
  else if(t.valid&&t.kind==='test')r.innerHTML='<span class="s">= '+esc(t.output)+'</span>';
  else r.innerHTML='<div class="rk">—</div>';
  row.appendChild(r); return row;
}
function mineTurnRow(t){
  const row=el('div','turn-row'); if(!t.valid&&!t.guess)row.classList.add('fmt');
  row.appendChild(el('div','t','#'+t.turn));
  const g=el('div','g');
  if(t.valid){ g.appendChild(el('div','w',t.guess)); if(t.mine_event==='boom')g.appendChild(el('div','e','지뢰 폭발 · 남은 목숨 '+t.lives)); else if(t.mine_event==='warn')g.appendChild(el('div','e','지뢰 접근 경보')); }
  else if(t.guess){ g.appendChild(el('div','w',t.guess)); g.appendChild(el('div','e','중복 추측')); }
  else { g.appendChild(el('div','w',((t.raw||'').replace(/\n/g,' ').trim().slice(0,22))||'(빈 응답)')); g.appendChild(el('div','e','형식 오류: GUESS 한 줄 필요')); }
  row.appendChild(g);
  const r=el('div','r');
  if(t.valid&&t.mine_event==='boom')r.innerHTML='<div class="rk" style="color:var(--crit);font-weight:800">폭발</div>';
  else if(t.valid&&t.rank!=null)r.innerHTML='<span class="s">유사도 '+(t.similarity*100).toFixed(1)+'%</span><div class="rk">'+t.rank+'위'+(t.mine_event==='warn'?' ⚠':'')+'</div>';
  else r.innerHTML='<div class="rk">—</div>';
  row.appendChild(r);
  if(t.valid&&t.rank!=null){ const bar=el('div','bar'); const i=el('i'); const c=closeness(t.rank); i.style.width=(c*100).toFixed(1)+'%'; i.style.background=heatColor(c); bar.appendChild(i); row.appendChild(bar); }
  return row;
}

/* ---------- 드로어(모델 상세) ---------- */
function openDrawer(model){ S.focus=model; $('#drawer').classList.add('open'); renderDrawer(); renderBoard(); }
function closeDrawer(){ S.focus=null; const d=$('#drawer'); if(d)d.classList.remove('open'); if(S.run)renderBoard(); }
function renderDrawer(){
  const model=S.focus; if(!model){return;}
  const gme=gameId(); if(MULTI.has(gme)){ renderDrawerG(gme,model); return; }
  const v=viewOf(model,S.episode);
  const dw=$('#drawer'); dw.innerHTML='';
  dw.style.setProperty('--mc',`var(${S.colorOf[model]})`);
  const head=el('div','dw-head');
  const top=el('div','top');
  top.appendChild(el('span','keydot'));
  top.appendChild(el('span','alias',aliasOf(model)));
  const deff=effortOf(model); if(deff) top.appendChild(el('span','eff',deff));
  const drb=reusedBadge(model); if(drb) top.appendChild(drb);
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
// picked: Map<modelId, Set<effort>>. 프리셋 없음 — 저장값이 있으면 복원, 없으면 빈 Map으로 시작.
// 저장/복원은 카탈로그로 검증(없는 모델·미지원 effort는 걸러 카탈로그 변경에도 안전). 파싱 실패 → 빈 Map.
const PICKED_KEY='arena.picked.v1';
function restorePicked(){
  const m=new Map();
  let raw; try{ raw=localStorage.getItem(PICKED_KEY); }catch(e){ return m; }
  if(!raw) return m;
  let arr; try{ arr=JSON.parse(raw); }catch(e){ return m; }   // 파싱 실패 → 조용히 빈 Map
  if(!Array.isArray(arr)) return m;
  arr.forEach(pair=>{
    if(!Array.isArray(pair)||pair.length<2) return;
    const mm=CFG.models.find(x=>x.id===pair[0]); if(!mm) return;      // 카탈로그에 없는 모델 제외
    if(!Array.isArray(pair[1])) return;
    const sup=(mm.efforts||CFG.efforts);
    const s=new Set(pair[1].filter(e=>sup.includes(e)));              // 미지원 effort 제외
    if(s.size) m.set(mm.id,s);
  });
  return m;
}
function savePicked(){
  try{ localStorage.setItem(PICKED_KEY,JSON.stringify([...picked].map(([id,s])=>[id,[...s]]))); }catch(e){}
}
let picked=restorePicked();
function partCount(){ let n=0; picked.forEach(s=>n+=s.size); return n; }
/* 게임 선택(런처): 세그먼트 · 설명 · 기본 max_turns 갱신 */
let selGame='ko-semantle';
let fillMode=false;   // 채우기 모드: 측정 조건(게임·시드·판수·턴수) 잠금 — 조건이 그룹과 같아야 재활용
function gameNoteText(){ const g=gmeta(selGame); return '게임: '+(g.kr||selGame)+' ('+selGame+')'; }
function buildGameSeg(){
  const seg=$('#gameSeg'); if(!seg) return; seg.innerHTML='';
  (CFG.games||['ko-semantle']).forEach(gid=>{
    const gm=gmeta(gid);
    const b=el('button','gseg'+(gid===selGame?' on':''));
    b.appendChild(el('span','gk',gm.kr||gid));
    b.appendChild(el('span','gid',gid));
    b.onclick=()=>{ if(fillMode) return; selGame=gid; buildGameSeg(); };   // 채우기 모드에선 게임 잠금
    seg.appendChild(b);
  });
  syncGame();
}
function syncGame(){
  const gm=gmeta(selGame);
  const d=$('#gameDesc'); if(d)d.textContent=gm.desc||'';
  const note=$('#newNote'); if(note)note.textContent=gameNoteText();
  const turns=$('#fTurns'); if(turns&&gm.max_turns) turns.value=gm.max_turns;
}
function openNew(){ exitFillMode(); buildGameSeg(); buildBulkEff(); buildModelGrid(); syncCount(); $('#newOverlay').classList.add('open'); }
function toggleEffort(mid,e){
  let set=picked.get(mid);
  if(set && set.has(e)){ set.delete(e); if(set.size===0) picked.delete(mid); }
  else{ if(!set){set=new Set();picked.set(mid,set);} set.add(e); }   // 상한 없음 — 선택은 막지 않는다
  buildModelGrid(); syncCount();
}
/* 모델 라벨 클릭 = 그 모델의 전 effort 토글(같은 모델을 effort별로 비교하는 워크플로).
   전부 켜져 있으면 해제, 아니면 지원 effort를 상한까지 추가. */
function toggleAllEfforts(mid){
  const m=CFG.models.find(x=>x.id===mid); const efs=(m&&m.efforts)||CFG.efforts;
  let set=picked.get(mid);
  if(set && set.size===efs.length){ picked.delete(mid); }   // 전부 켜짐 → 해제
  else{ if(!set){ set=new Set(); picked.set(mid,set); } efs.forEach(e=>set.add(e)); }   // 전 effort 켜기(상한 없음)
  buildModelGrid(); syncCount();
}
function selectNoneModels(){ picked=new Map(); buildModelGrid(); syncCount(); }
// 그룹 [전체] 켜짐 판정: 그룹 전 모델이 각자 지원하는 전 effort를 켠 상태(부분 선택이면 꺼짐 — 정직).
function familyAllOn(fam){
  return fam.models.every(m=>{ const efs=(m.efforts||CFG.efforts); const s=picked.get(m.id);
    return !!(s && efs.every(e=>s.has(e))); });
}
/* 그룹 [전체] 토글 — 모델 라벨 클릭(전 effort)의 그룹판(참가자 = 측정 키와 무관 → fillMode에서도 동작).
   켜기: 그룹 전 모델 × 각자 지원하는 전 effort 선택. 끄기: 그룹 전 모델 해제. 상한 없음. */
function toggleFamily(fam){
  if(familyAllOn(fam)){ fam.models.forEach(m=>picked.delete(m.id)); }
  else{ fam.models.forEach(m=>{ const efs=(m.efforts||CFG.efforts);
    let s=picked.get(m.id); if(!s){ s=new Set(); picked.set(m.id,s); } efs.forEach(e=>s.add(e)); }); }
  buildModelGrid(); syncCount();
}
/* 전 모델 effort 칩 줄 생성: CFG.efforts(개별 칩과 동일 소스)에서 5칩.
   상태 있는 토글 — 켜짐(.on) = 지원하는 전 모델이 그 effort를 갖고 있음. */
function buildBulkEff(){
  const host=$('#bulkEff'); if(!host) return;
  host.querySelectorAll('.echip').forEach(c=>c.remove());   // 라벨·메시지 유지, 칩만 재생성
  const anchor=$('#bulkEffMsg');
  (CFG.efforts||['low','medium','high','xhigh','max']).forEach(e=>{
    const cb=el('button','echip'+(bulkEffState(e)?' on':''),e); cb.type='button';
    cb.title='전 모델에 '+e+' 켜기/끄기';
    cb.onclick=()=>toggleBulkEffort(e);
    host.insertBefore(cb,anchor);
  });
}
// 모델이 effort e를 지원하는가 — CFG.models의 efforts 목록 기준(하드코딩 금지).
function supportsEffort(id,e){ const m=CFG.models.find(x=>x.id===id); return ((m&&m.efforts)||CFG.efforts).includes(e); }
let _bulkMsgT=null;
function bulkEffMsg(t){ const m=$('#bulkEffMsg'); if(!m) return; m.textContent=t;
  clearTimeout(_bulkMsgT); _bulkMsgT=setTimeout(()=>{ m.textContent=''; },4000); }
// 토글 상태: e를 지원하는 전 모델이 e를 켜둔 상태인가(상한으로 일부만 적용되면 꺼짐으로 표시 — 정직).
function bulkEffState(e){
  const sup=CFG.models.filter(m=>((m.efforts)||CFG.efforts).includes(e));
  return sup.length>0 && sup.every(m=>{ const s=picked.get(m.id); return !!(s && s.has(e)); });
}
/* 전 모델 effort 토글(참가자 자유 → fillMode에서도 동작). 여러 칩을 동시에 켤 수 있다. 상한 없음.
   켜기: 지원 모델 전부에 e 추가 — 기존 선택·다른 effort 보존.
   끄기: 전 모델에서 e 제거 — 빈 선택이 된 모델은 해제. 미지원 모델은 어느 쪽도 건드리지 않음. */
function toggleBulkEffort(e){
  if(bulkEffState(e)){
    let removed=0;
    picked.forEach((set,id)=>{ if(set.has(e)){ set.delete(e); removed++; if(set.size===0) picked.delete(id); } });
    bulkEffMsg(removed+'개 모델에서 '+e+' 해제');
  }else{
    let added=0;
    CFG.models.forEach(m=>{
      if(!supportsEffort(m.id,e)) return;
      let set=picked.get(m.id);
      if(set && set.has(e)) return;
      if(!set){ set=new Set(); picked.set(m.id,set); }
      set.add(e); added++;
    });
    bulkEffMsg(added+'개 모델에 '+e+' 추가');
  }
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
    // 그룹 [전체] 토글: 전 모델×전 effort 켜짐 시 '해제', 아니면 그룹 전 모델×전 effort 선택.
    const allOn=familyAllOn(fam);
    const ft=el('button','fam-toggle'+(allOn?' on':''),'전체'); ft.type='button';
    ft.title=allOn?'이 그룹 전 모델×전 effort 해제':'이 그룹 전 모델×전 effort 선택';
    ft.onclick=()=>toggleFamily(fam);
    head.appendChild(ft);
    g.appendChild(head);
    fam.models.forEach(m=>{
      const set=picked.get(m.id);
      const on=!!(set && set.size);
      const opt=el('div','mopt'+(on?' on':'')); opt.title=m.id;  // 전체 id는 title 툴팁만
      const al=el('span','al',m.ver);                            // 버전 라벨 하나만
      al.title=m.id+' — 클릭: 전 effort 토글';
      al.onclick=()=>toggleAllEfforts(m.id);
      opt.appendChild(al);
      const er=el('div','effrow');
      (m.efforts||CFG.efforts).forEach(e=>{
        const active=!!(set && set.has(e));
        const cb=el('button','echip'+(active?' on':''),e);
        cb.onclick=()=>toggleEffort(m.id,e);
        er.appendChild(cb);
      });
      opt.appendChild(er);
      g.appendChild(opt);
    });
  });
  buildBulkEff();   // 개별 칩 조작에도 전 모델 effort 토글의 켜짐 상태를 동기화
}
function syncCount(){
  const n=partCount();
  const c=$('#mcount'); c.innerHTML='<b>'+n+'</b> 참가자';   // 상한 없음 — 분모(/ N) 표기 제거
  $('#startBtn').disabled=n===0;
  $('#startBtn').style.opacity=n===0?.5:1;
  savePicked();   // 모든 선택 변경 경로가 지나는 지점 — 여기서 저장하면 누락 없이 이어하기 가능
}
function currentParticipants(){
  const out=[];
  CFG.models.forEach(m=>{ const set=picked.get(m.id); if(set) (m.efforts||CFG.efforts).forEach(e=>{ if(set.has(e)) out.push({model:m.id,effort:e}); }); });
  return out;
}
/* 측정 경제 토글: 시각 상태(.on) 동기화 */
function syncEcon(){
  const r=$('#remeasureRow'); if(r) r.classList.toggle('on',$('#fRemeasure').checked);
}
$('#fRemeasure').addEventListener('change',syncEcon);
// 조건 잠금 해제: 채우기 모드 → 일반 모드(측정 조건 전부 활성 · 배너 제거). 참가자는 그대로 유지.
$('#fillUnlockBtn').onclick=exitFillMode;
// 참가자 일괄 선택(fillMode에서도 동작 — 참가자는 측정 키와 무관).
$('#selNoneBtn').onclick=selectNoneModels;
// 🎲 랜덤 시드: 이제 랜덤은 기본이 아니라 사용자의 명시적 선택(0~2^31).
$('#seedDice').onclick=()=>{ $('#fSeed').value=Math.floor(Math.random()*0x80000000); $('#newNote').textContent=gameNoteText(); };
$('#startBtn').onclick=async()=>{
  const participants=currentParticipants();
  if(participants.length===0) return;
  // 시드: 항상 명시적으로 전송(같은 시드 = 같은 문제 세트 → 완주 결과 자동 재활용).
  // 빈 값·비정수·음수·표현 불가한 큰 수는 시작을 막고 정직히 안내.
  const seedRaw=($('#fSeed').value||'').trim();
  if(!/^\d+$/.test(seedRaw)){ $('#newNote').textContent='시드는 0 이상의 정수여야 합니다'; return; }
  const seed=parseInt(seedRaw,10);
  if(!Number.isSafeInteger(seed)||seed<0){ $('#newNote').textContent='시드는 0 이상의 정수여야 합니다'; return; }
  const remeasure=$('#fRemeasure').checked;   // '전부 다시 측정' → reuse:false
  // episodes = 반복 수. repeat_seed 항상 true → 엔진이 같은 문제(시드)를 N회 반복.
  // (1회면 seeds=[S] — 오늘까지의 1에피소드 런과 동일 조건·측정 키이므로 커버리지가 이어진다.)
  const workers=+$('#fWorkers').value||CFG.defaultWorkers||32;   // 최대 동시 실행(측정 키와 무관)
  const body={game:selGame,participants,
    episodes:+$('#fEpisodes').value||1, max_turns:+$('#fTurns').value||1,
    reuse:!remeasure, seed, repeat_seed:true, workers};
  // 스폰 전 기존 run_id 집합을 기억 — 새로 생긴 런만 골라 선택한다(기존 최신 런 오선택 방지).
  const before=new Set(((S.index&&S.index.runs)||[]).map(r=>r.run_id));
  $('#startBtn').disabled=true; $('#newNote').textContent='시작 요청 중…';
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok){ $('#newNote').textContent='오류: '+(d.error||r.status); $('#startBtn').disabled=false; return; }
    // 성공 응답 직후 오버레이를 닫는다(닫기 타이밍 유지). 준비/진행 상태는 이후
    // 상단 상태칩과 보드('게임 준비 중')가 정직하게 보여준다.
    $('#newNote').textContent='게임 준비 중…';
    close('newOverlay'); $('#startBtn').disabled=false; S.runId=null;
    // 엔진이 예비 manifest(status:preparing)를 쓰면 새 런이 index에 뜬다. 오라클 콜드
    // 로딩이 수십 초 걸릴 수 있어 넉넉히 폴링(45×1s). 뜨는 즉시 자동 선택한다.
    if(await awaitNewRun(before,45)) return;
    // 시간 초과: 조용한 포기 금지 — 정직한 안내 + 백그라운드 폴링 유지(뜨면 자동 편입).
    $('#newNote').textContent='준비가 오래 걸리고 있습니다 — 준비가 끝나면 자동으로 나타납니다';
    keepAwaitingNewRun(before);
  }catch(e){ $('#newNote').textContent='요청 실패'; $('#startBtn').disabled=false; }
};
// 스폰된 새 런(before에 없던 run_id)이 index에 나타날 때까지 폴링해 선택. 발견 시 true.
async function awaitNewRun(before,tries){
  for(let i=0;i<tries;i++){
    await new Promise(rs=>setTimeout(rs,1000));
    if(S.runId) return true;                      // 사용자가 그 사이 다른 런을 열었으면 중단
    try{ await loadIndex(false); }catch(e){ continue; }
    const fresh=((S.index&&S.index.runs)||[]).find(r=>!before.has(r.run_id));
    if(fresh){ selectRun(fresh.run_id); return true; }
  }
  return false;
}
// 시간 초과 후에도 조용히 포기하지 않고 저속 폴링을 유지 — 준비가 끝나 런이 뜨면 자동 선택.
function keepAwaitingNewRun(before){
  if(S._waitRun) return;
  S._waitRun=setInterval(async()=>{
    if(S.runId){ clearInterval(S._waitRun); S._waitRun=null; return; }
    try{ await loadIndex(false); }catch(e){ return; }
    const fresh=((S.index&&S.index.runs)||[]).find(r=>!before.has(r.run_id));
    if(fresh){ clearInterval(S._waitRun); S._waitRun=null; selectRun(fresh.run_id); }
  },3000);
}

/* ---------- 기록 ---------- */
async function openHist(){
  $('#histOverlay').classList.add('open');
  await Promise.all([renderHistSeeds(), renderHistRuns()]);
  buildHistModelSel();
  const t=localStorage.getItem('arena-hist-tab');
  showHistTab(['seeds','runs','models'].includes(t)?t:'seeds');   // 기본 시드별(행동 가능한 뷰)
}
function showHistTab(which){
  const panes={seeds:'#histSeeds',runs:'#histRuns',models:'#histModels'};
  const tabs={seeds:'#tabSeeds',runs:'#tabRuns',models:'#tabModels'};
  for(const k in panes){ $(panes[k]).classList.toggle('hidden',k!==which); $(tabs[k]).classList.toggle('on',k===which); }
  try{ localStorage.setItem('arena-hist-tab',which); }catch(e){}
}
function defaultEffortOf(id){ const m=CFG.models.find(x=>x.id===id); const efs=(m&&m.efforts)||['low']; return efs.includes('low')?'low':efs[0]; }
function histStatusChip(status){
  const preparing=(status==='preparing'), running=(status==='running'), failed=(status==='failed'), stopped=(status==='stopped');
  return el('span','rc-st st '+(preparing?'preparing':running?'running':failed?'failed':stopped?'stopped':'done'),
    preparing?'준비 중':running?'LIVE':failed?'중단':stopped?'정지됨':'완료');
}
function fmtWhen(iso){ return iso? String(iso).replace('T',' ').slice(0,16) : ''; }

/* ---------- 기록: 시드별(측정 조건 단위 coverage) ---------- */
async function renderHistSeeds(){
  const box=$('#histSeeds'); box.innerHTML='<div class="hint-load">불러오는 중…</div>';
  let d; try{ d=await jget('/api/seeds'); }catch(e){ box.innerHTML='<div class="hint-load">불러오지 못했습니다.</div>'; return; }
  box.innerHTML='';
  const groups=d.groups||[];
  if(!groups.length){ box.innerHTML='<div class="hint-load">기록이 없습니다.</div>'; return; }
  // 시드 우산으로 재그룹(서버가 최근 활동 순으로 준 groups의 첫 등장 순서 보존).
  const bySeed=new Map(); const order=[];
  groups.forEach(g=>{ const k=(g.seed==null?'__none__':String(g.seed));
    if(!bySeed.has(k)){ bySeed.set(k,[]); order.push(k); } bySeed.get(k).push(g); });
  const gorder=CFG.games||['ko-semantle'];
  order.forEach(k=>{
    const gs=bySeed.get(k); const seed=gs[0].seed;
    const um=el('div','seedgrp');
    const head=el('div','sg-head');
    head.appendChild(el('span','sg-seed', seed==null?'시드 기록 없음':('시드 '+seed)));
    if(seed!=null && CFG.suiteSeed!=null && seed===CFG.suiteSeed) head.appendChild(el('span','sg-tag','정규 세트'));
    um.appendChild(head);
    gs.sort((a,b)=> (gorder.indexOf(a.game)-gorder.indexOf(b.game))
      || ((a.max_turns||0)-(b.max_turns||0)) || ((a.episodes||0)-(b.episodes||0)));
    gs.forEach(g=>um.appendChild(seedGameRow(g)));
    if(seed!=null){   // 이 시드에 런 없는 게임 → [이 시드로 시작] 한 줄씩
      const have=new Set(gs.map(g=>g.game));
      gorder.filter(gid=>!have.has(gid)).forEach(gid=>um.appendChild(unrunGameRow(gid,seed)));
    }
    box.appendChild(um);
  });
}
function coverageOf(g){
  const catalog=CFG.models.map(m=>m.id);
  const measuredModels=new Set((g.measured||[]).map(s=>String(s).split('@')[0]));
  const measured=catalog.filter(id=>measuredModels.has(id));
  const unmeasured=catalog.filter(id=>!measuredModels.has(id));
  return {catalog, measured, unmeasured, measuredModels};
}
function seedGameRow(g){
  const row=el('div','sg-row');
  const top=el('div','sgr-top');
  const gm=gmeta(g.game);
  top.appendChild(el('span','sgr-game', gm.kr||g.game));
  top.appendChild(el('span','sgr-cond','· '+seedCondLabel(g)));   // 반복/다문제/단판 정직 표기
  const cov=coverageOf(g);
  const covc=el('span','sgr-cov'); covc.innerHTML='측정 <b>'+cov.measured.length+'</b>/'+cov.catalog.length+'모델';
  top.appendChild(covc);
  const btns=el('div','sgr-btns');
  if(cov.unmeasured.length){
    const b=el('button','minibtn fill','미측정 채우기');
    b.onclick=(e)=>{ e.stopPropagation(); fillUnmeasured(g); };
    btns.appendChild(b);
  }
  if(g.latest_run){
    const b=el('button','minibtn','최신 런 열기');
    b.onclick=(e)=>{ e.stopPropagation(); close('histOverlay'); selectRun(g.latest_run); };
    btns.appendChild(b);
  }
  top.appendChild(btns);
  top.onclick=()=>row.classList.toggle('open');   // 펼침
  row.appendChild(top);
  const det=el('div','sgr-det');
  if((g.measured||[]).length){
    const mr=el('div','sgr-chips'); mr.appendChild(el('span','sgr-lab','측정됨'));
    g.measured.forEach(s=>mr.appendChild(el('span','mchip',slugLabel(s)))); det.appendChild(mr);
  }
  if(cov.unmeasured.length){
    const ur=el('div','sgr-chips'); ur.appendChild(el('span','sgr-lab','미측정'));
    cov.unmeasured.forEach(id=>ur.appendChild(el('span','uchip',nameOf(id)))); det.appendChild(ur);
  }
  const rl=el('div','sgr-runs');
  (g.runs||[]).forEach(r=>{
    const rr=el('div','sgr-run');
    rr.appendChild(el('span','rid',String(r.run_id).replace(/^arena-/,'')));
    rr.appendChild(histStatusChip(r.status));
    rr.appendChild(el('span','when',fmtWhen(r.started_at)));
    rr.onclick=(e)=>{ e.stopPropagation(); close('histOverlay'); selectRun(r.run_id); };
    rl.appendChild(rr);
  });
  det.appendChild(rl);
  row.appendChild(det);
  return row;
}
function unrunGameRow(gid,seed){
  const row=el('div','sg-row unrun');
  const top=el('div','sgr-top');
  const gm=gmeta(gid);
  top.appendChild(el('span','sgr-game muted', gm.kr||gid));
  top.appendChild(el('span','sgr-cond','· 이 시드로 미측정'));
  const spacer=el('span','sgr-cov'); top.appendChild(spacer);
  const b=el('button','minibtn start','이 시드로 시작');
  b.onclick=(e)=>{ e.stopPropagation(); startWithSeed(gid,seed); };
  const btns=el('div','sgr-btns'); btns.appendChild(b); top.appendChild(btns);
  row.appendChild(top);
  return row;
}
// 채우기 모드 잠금 적용/해제: 측정 조건 입력(게임·시드·판수·턴수·전부 다시 측정)만 잠근다.
// 참가자 선택은 측정 키와 무관 → 자유. fillMode 값을 읽어 disabled·잠금 클래스·배너를 동기화.
const FILL_LOCK_IDS=['fSeed','seedDice','fEpisodes','fTurns','fRemeasure'];
function applyFillLock(){
  FILL_LOCK_IDS.forEach(id=>{ const e=$('#'+id); if(e) e.disabled=fillMode; });
  const seg=$('#gameSeg'); if(seg) seg.classList.toggle('locked',fillMode);
  const rm=$('#remeasureRow'); if(rm) rm.classList.toggle('locked',fillMode);
  const banner=$('#fillBanner'); if(banner) banner.hidden=!fillMode;
}
function exitFillMode(){ fillMode=false; applyFillLock(); }   // 일반 모드로: 전부 활성 · 배너 제거
// 미측정 채우기: 기존 측정 슬러그 전부(자동 재활용) + 미측정 모델(기본 effort)을 런처에 프리필.
// 채우기 모드(fill:true)로 열어 측정 조건을 잠근다 — 조건이 그룹과 같아야 완주분이 재활용된다.
function fillUnmeasured(g){
  const cov=coverageOf(g);
  const participants=[];
  (g.measured||[]).forEach(s=>{ const a=String(s).split('@'); participants.push({model:a[0], effort:a[1]||defaultEffortOf(a[0])}); });
  cov.unmeasured.forEach(id=>participants.push({model:id, effort:defaultEffortOf(id)}));
  openNewPrefill({game:g.game, seed:g.seed, episodes:g.episodes, max_turns:g.max_turns,
    participants, fill:true});
}
function startWithSeed(gid,seed){
  // 미측정 시드 시작: 게임·시드만 프리필, 조건·참가자는 자유(재활용할 완주분이 없어 잠그지 않음).
  openNewPrefill({game:gid, seed, participants:[], note:'시드 '+seed+' — 참가자를 선택해 시작하세요'});
}
// 런처 오버레이를 값으로 프리필해 연다(자동 발사 금지 — 사용자가 시작 버튼으로 비용 확인).
function openNewPrefill(opt){
  close('histOverlay');
  if(opt.game) selGame=opt.game;
  buildGameSeg();                         // selGame UI(활성 버튼)·게임 설명·하단 '게임: …' 문구 동기화
  if(opt.seed!=null) $('#fSeed').value=opt.seed;
  if(opt.episodes!=null) $('#fEpisodes').value=opt.episodes;
  if(opt.max_turns!=null) $('#fTurns').value=opt.max_turns;
  if(opt.participants){
    picked=new Map();
    opt.participants.forEach(p=>{   // 상한 없음 — 프리필 전부 반영
      let s=picked.get(p.model); if(!s){ s=new Set(); picked.set(p.model,s); } s.add(p.effort);
    });
  }
  buildBulkEff(); buildModelGrid(); syncCount();
  if(opt.fill){
    // 채우기 모드: 컨텍스트 배너에 측정 조건 명시(반복 수 포함 — 병합 행은 최신 런 조건으로 채운다).
    // 하단 문구는 buildGameSeg가 세운 '게임: …' 유지.
    const gm=gmeta(selGame);
    const eps=(opt.episodes!=null?opt.episodes:1);
    const cond=playLabel(eps)+'×'+(opt.max_turns!=null?opt.max_turns:(gm.max_turns||''))+'턴';
    $('#fillBannerText').textContent='시드 '+opt.seed+' · '+(gm.kr||selGame)+' · '+cond
      +' 채우기 — 조건이 같아야 기존 완주 결과가 재활용됩니다(측정분은 비용 없이 함께 표시)';
    fillMode=true;
  }else{
    fillMode=false;
    if(opt.note) $('#newNote').textContent=opt.note;   // 비-채우기: 안내 문구(syncGame이 덮으므로 마지막에)
  }
  applyFillLock();
  $('#newOverlay').classList.add('open');
}
// 실행 중인 런 정지: confirm 한 번 → POST /api/stop → 목록·상세 갱신(라이브 폴링은 stopped로 자연 중단).
async function stopRun(runId){
  if(!confirm('이 런을 정지할까요? 진행 중인 플레이가 중단됩니다.')) return;
  try{
    const r=await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:runId})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok){ alert('정지 실패: '+((d&&d.error)||r.status)); return; }
  }catch(e){ alert('정지 요청 실패'); return; }
  try{ await loadIndex(false); }catch(e){}                       // 목록 갱신
  if(S.runId===runId){ try{ await selectRun(runId); }catch(e){} }   // 상세 갱신
  try{ renderHistRuns(); }catch(e){}
  try{ renderHistSeeds(); }catch(e){}                            // 시드별 커버리지도 반영
}
async function renderHistRuns(){
  const box=$('#histRuns'); box.innerHTML='';
  const runs=(S.index&&S.index.runs)||[];
  runs.forEach(r=>{
    const card=el('div','runcard');
    // 상단: 런ID · 게임 배지 · 상태칩(index status 정직 반영)
    const head=el('div','rc-head');
    head.appendChild(el('span','rc-rid',String(r.run_id).replace(/^arena-/,'')));
    head.appendChild(el('span','rc-game',gmeta(r.game||'ko-semantle').kr||r.game||''));
    // preparing을 반드시 먼저 판정 — 아니면 running/failed 외 전부 '완료'로 오표기된다(정직성 위반).
    const preparing=(r.status==='preparing'), running=(r.status==='running'), failed=(r.status==='failed'), stopped=(r.status==='stopped');
    head.appendChild(el('span','rc-st st '+(preparing?'preparing':running?'running':failed?'failed':stopped?'stopped':'done'),
      preparing?'준비 중':running?'LIVE':failed?'중단':stopped?'정지됨':'완료'));
    if(running){   // 실행 중인 런만 정지 가능 — confirm 후 정지 API
      const sb=el('button','minibtn stop','정지'); sb.title='이 런을 정지';
      sb.onclick=(e)=>{ e.stopPropagation(); stopRun(r.run_id); };
      head.appendChild(sb);
    }
    card.appendChild(head);
    // 1등 줄: 풀네임 · effort — 업적(없으면 미해결)
    const win=el('div','rc-win');
    win.appendChild(el('span','rc-crown','1등'));
    const w=r.winner;
    if(w){
      win.appendChild(el('span','rc-wname',nameOf(w.model)));
      if(w.effort) win.appendChild(el('span','eff',w.effort));
      if(w.achievement) win.appendChild(el('span','rc-ach',w.achievement));
    }else{
      win.appendChild(el('span','rc-none','미해결'));
    }
    card.appendChild(win);
    // 메타: 참가 수 · 에피소드 · 최대 턴
    const pc=(r.participants_count!=null)?r.participants_count:(r.models||[]).length;
    card.appendChild(el('div','rc-meta',`참가 ${pc} · ${r.episodes}ep · ${r.max_turns}턴`));
    card.onclick=()=>{ close('histOverlay'); selectRun(r.run_id); };
    box.appendChild(card);
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
  // (게임·시드·턴·effort) 조건으로 묶어 반복/재실행을 플레이 가중 평균한다("N플레이 평균").
  // 같은 (게임·시드·턴)의 1판 런 1회 + 4회 반복 런 4회 = 5표본으로 합산·평균. effort는 별 행(비교 유지).
  const groups=new Map();
  (d.runs||[]).forEach(r=>(r.participants||[]).forEach(p=>{
    const key=[r.game,r.seed,r.max_turns,p.effort].join('|');
    let g=groups.get(key);
    if(!g){ g={game:r.game,seed:r.seed,max_turns:r.max_turns,effort:p.effort,items:[],latest:null,latestAt:''}; groups.set(key,g); }
    g.items.push({r,p});
    const at=r.started_at||''; if(at>=g.latestAt){ g.latestAt=at; g.latest=r.run_id; }
  }));
  const list=[...groups.values()].sort((a,b)=> (b.latestAt>a.latestAt?1:b.latestAt<a.latestAt?-1:0));
  if(!list.length){box.innerHTML='<div style="color:var(--muted);font-size:12px">참여한 런이 없습니다.</div>';return;}
  const num=x=>typeof x==='number'&&isFinite(x);
  list.forEach(g=>{
    // 플레이 가중 평균: 각 런의 요약(그 런 에피소드 집계)을 반복 횟수로 가중 → 전 플레이 대평균.
    let plays=0, runsN=0; const acc={score:[0,0], solve:[0,0], turn:[0,0]};   // [Σ(v*w), Σw]
    g.items.forEach(({r,p})=>{
      const s=p.summary; if(!s) return;
      const w=num(r.episodes)?r.episodes:1; plays+=w; runsN++;
      if(num(s.mean_score)){ acc.score[0]+=s.mean_score*w; acc.score[1]+=w; }
      if(num(s.solve_rate)){ acc.solve[0]+=s.solve_rate*w; acc.solve[1]+=w; }
      if(num(s.median_turns)){ acc.turn[0]+=s.median_turns*w; acc.turn[1]+=w; }
    });
    const avg=a=>a[1]>0?a[0]/a[1]:null;
    const row=el('div','mrunrow');
    const left=el('div');
    const rl=el('div','rid'); rl.appendChild(document.createTextNode(g.seed!=null?('시드 '+g.seed):'시드 없음'));
    if(g.effort) rl.appendChild(el('span','eff',g.effort));
    left.appendChild(rl);
    // 표본 수 명시(라벨 없는 합산 금지): "N플레이 평균"(런 여러 개면 "· M런").
    const sampLab=(plays>0?(plays+'플레이 평균'):'플레이 없음')+(runsN>1?(' · '+runsN+'런'):'');
    left.appendChild(el('div','ms', g.max_turns+'턴 · '+sampLab));
    row.appendChild(left);
    const sv=avg(acc.solve), sc=avg(acc.score), tn=avg(acc.turn);
    const met=el('div','metrics');
    met.appendChild(metric(sv!=null?(sv*100).toFixed(0)+'%':'—','해결률'));
    met.appendChild(metric(sc!=null?(sc*100).toFixed(0):'—','점수'));
    met.appendChild(metric(tn!=null?(Math.round(tn*10)/10):'—','중앙턴'));
    row.appendChild(met);
    row.onclick=()=>{ if(g.latest){ close('histOverlay'); selectRun(g.latest); } };
    box.appendChild(row);
  });
}
function metric(v,k){const m=el('div');m.appendChild(el('div','v',v));m.appendChild(el('div','k',k));return m;}
$('#tabSeeds').onclick=()=>showHistTab('seeds');
$('#tabRuns').onclick=()=>showHistTab('runs');
$('#tabModels').onclick=()=>showHistTab('models');

/* ---------- 오버레이 공통 ---------- */
// 런처를 닫으면 채우기 모드도 함께 해제(다음 일반 열기로 잠금 상태가 누출되지 않게).
function close(id){ $('#'+id).classList.remove('open'); if(id==='newOverlay') exitFillMode(); }
$$('[data-close]').forEach(b=>b.onclick=()=>close(b.dataset.close));
$$('.overlay').forEach(o=>o.addEventListener('click',e=>{if(e.target===o){o.classList.remove('open'); if(o.id==='newOverlay') exitFillMode();}}));
$('#newBtn').onclick=openNew;
$('#emptyNewBtn').onclick=openNew;
$('#histBtn').onclick=openHist;
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$$('.overlay.open').forEach(o=>o.classList.remove('open'));exitFillMode();if(S.focus)closeDrawer();}});

/* ---------- 리사이즈: 궤적 SVG 재측정 ---------- */
let _trendRAF=null;
window.addEventListener('resize',()=>{
  if(!S.run) return;
  if(_trendRAF) cancelAnimationFrame(_trendRAF);
  _trendRAF=requestAnimationFrame(renderBoard);
});

/* ---------- 부트 ---------- */
initTheme();
buildBulkEff();   // 일괄 effort 칩(상태 없는 액션 → 1회 생성)
// 시드 입력 기본값을 CFG.suiteSeed(정규 문제 세트 고정 seed)로 프리필 — 기본은 재활용.
if($('#fSeed') && CFG.suiteSeed!=null) $('#fSeed').value=CFG.suiteSeed;
// 최대 동시 실행 기본값 = 서버 MAX_WORKERS_DEFAULT.
if($('#fWorkers') && CFG.defaultWorkers!=null) $('#fWorkers').value=CFG.defaultWorkers;
loadIndex(true);
</script>
</body>
</html>
"""
