"""Mindmatch 실행 엔진 — 참가자(모델×effort) 동시 플레이 + 즉시 저장 + 재생 검증.

설계 원칙(R4~R7):
- 참가자 단위는 '모델×effort 조합'이다: slug = "<model>@<effort>".
  저장·조회 단위가 slug다: results/arena/<run>/models/<slug>/.
- 모든 참가자는 에피소드별 동일 seed(=동일 정답 단어)를 받아 공정 비교한다(R4).
- 매 턴 events.jsonl append + live.json 원자 교체 → 중단에 강건(R7 저장 즉시성).
- LLM 심판 없음, 종합 우승자 점수 없음(R7). 참가자별 지표만 저장한다.
- 정답 누출 금지: target은 episode_end 이벤트에만. live.json·turn 이벤트엔 절대 없음.

파일 계약(v2, 웹 쪽과 공유하는 접점 — 변경 금지):
  results/arena/index.json
  results/arena/<run_id>/manifest.json
    - participants: [{"model","effort","slug"}]
    - models: slug 리스트(index 표시용), effort: 기본 effort
  results/arena/<run_id>/models/<slug>/live.json      (원자 교체, model=순수id + effort)
  results/arena/<run_id>/models/<slug>/events.jsonl   (턴마다 append, 스키마 v1 그대로)
  results/arena/<run_id>/models/<slug>/summary.json   (모델 완료 시, model=순수id + effort)

레거시 허용: participants 없는 구형 런(디렉토리=모델 id)은 manifest.models×manifest.effort
로 유도해 계속 verify_run이 동작한다.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import client, config, embed
from .games import build_game


RAW_LIMIT = 600           # live.json 표시용 스니펫 절단 길이(events.jsonl에는 전문 저장)
MAX_WORKERS_DEFAULT = 12  # 동시 실행 상한 기본값(조합 폭주로 CLI 프로세스가 터지는 것 방지)
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_INDEX_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# 참가자(모델×effort) 정규화
# ----------------------------------------------------------------------
def _parse_participant(item, default_effort: str) -> dict:
    """참가자 항목 → {"model","effort","slug"}.

    item은 {"model","effort"} dict 또는 "model@effort"/"model" 문자열을 수용한다.
    effort 미지정 항목은 default_effort를 쓴다.
    """
    if isinstance(item, dict):
        model = str(item["model"]).strip()
        effort = str(item.get("effort") or default_effort).strip()
    else:
        text = str(item).strip()
        if "@" in text:
            model, _, effort = text.partition("@")
            model, effort = model.strip(), (effort.strip() or default_effort)
        else:
            model, effort = text, default_effort
    # (1) 오타 검사: 애초에 존재하는 effort 단계인가.
    if effort not in VALID_EFFORTS:
        raise ValueError(
            f"unknown effort: {effort!r} — 오타로 보입니다. "
            f"유효 effort: {', '.join(VALID_EFFORTS)}")
    # (2) 미지원 검사: 이 모델이 실제 지원하는 단계인가(측정 무결성).
    #     codex엔 max가 없고 gemini는 High 고정 — 미지원을 조용히 강등/무시하면
    #     같은 조건이 다른 레인으로 위장된다.
    allowed = config.model_efforts(model)
    if effort not in allowed:
        raise ValueError(
            f"effort {effort!r}는 {model}에서 지원하지 않습니다 — "
            f"허용: {', '.join(allowed)}")
    return {"model": model, "effort": effort, "slug": f"{model}@{effort}"}


def _normalize_participants(participants, default_effort: str) -> list[dict]:
    """(model,effort) 중복 제거, 입력 순서 보존."""
    out, seen = [], set()
    for item in participants:
        p = _parse_participant(item, default_effort)
        key = (p["model"], p["effort"])
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# ----------------------------------------------------------------------
# 저수준 IO 헬퍼
# ----------------------------------------------------------------------
def _stamp() -> str:
    return "arena-" + time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _write_json_atomic(path: Path, data: dict) -> None:
    """임시파일에 쓰고 os.replace로 원자 교체 → 부분쓰기 노출 차단."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_records(path: Path) -> list[dict]:
    records = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 정전 후 마지막 잘린 줄은 관용
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        pass
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ----------------------------------------------------------------------
# index.json — 런 목록(최신이 앞)
# ----------------------------------------------------------------------
def _index_entry(manifest: dict) -> dict:
    return {k: manifest.get(k) for k in (
        "run_id", "game", "models", "episodes", "max_turns",
        "effort", "status", "started_at", "finished_at")}


def _index_upsert(arena_root: Path, entry: dict) -> None:
    path = Path(arena_root) / "index.json"
    with _INDEX_LOCK:
        data = _read_json(path)
        runs = data.get("runs")
        runs = list(runs) if isinstance(runs, list) else []
        runs = [r for r in runs if r.get("run_id") != entry["run_id"]]
        runs.insert(0, entry)  # 최신이 앞
        _write_json_atomic(path, {"runs": runs})


def _model_dir(run_dir: Path, name: str) -> Path:
    """참가자 디렉토리(v2=slug, 레거시=모델 id)."""
    return Path(run_dir) / "models" / name


# ----------------------------------------------------------------------
# 이벤트/라이브 조립
# ----------------------------------------------------------------------
def _turn_event(episode: int, step_event: dict, best_rank) -> dict:
    """게임 step 이벤트 → events.jsonl 턴 레코드(스키마 v1 그대로).

    valid 턴은 guess/similarity/rank를, 무효 턴은 error를 싣는다. 중복 추측
    무효 턴은 게임 엔진이 guess를 그대로 남기므로(재생 대조용) 함께 보존한다.
    정답(target)은 어느 턴 이벤트에도 넣지 않는다.
    """
    ev = {"type": "turn", "episode": episode, "turn": step_event["turn"],
          "valid": bool(step_event.get("valid"))}
    if step_event.get("valid"):
        ev["guess"] = step_event["guess"]
        ev["similarity"] = step_event["similarity"]
        ev["rank"] = step_event["rank"]
        ev["sim_to_prev"] = step_event.get("sim_to_prev")  # 직전 유효 추측과의 코사인(첫 턴 null)
    else:
        if "guess" in step_event:  # 중복 추측 무효 턴은 guess를 보존
            ev["guess"] = step_event["guess"]
        ev["error"] = step_event.get("error", "invalid")
    ev["best_rank"] = best_rank
    # 전문 보존: 재생 검증이 이 raw를 다시 parse하므로 절단하면 오탐이 난다.
    ev["raw"] = step_event.get("raw") or ""
    ev["ts"] = _now()
    return ev


def _live(model: str, effort: str, episode: int, turn: int, max_turns: int,
          phase: str, event: dict | None, best_rank) -> dict:
    return {
        "model": model,     # 순수 모델 id(slug 아님)
        "effort": effort,
        "episode": episode,
        "turn": turn,
        "max_turns": max_turns,
        "phase": phase,
        "last_guess": (event or {}).get("guess", ""),
        "last_similarity": (event or {}).get("similarity"),
        "last_rank": (event or {}).get("rank"),
        "best_rank": best_rank,
        "raw_snippet": ((event or {}).get("raw", ""))[:RAW_LIMIT],
        "updated_at": _now(),
    }


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def _call(model: str, prompt: str, effort: str, timeout: int, retries: int = 1):
    """모델 1턴 호출(재시도 retries회). 실패 시 (None, error)."""
    last_err = ""
    for attempt in range(retries + 1):
        resp = client.call(model, prompt, effort=effort, timeout=timeout)
        if resp.ok:
            return resp, ""
        last_err = resp.error or "model_error"
        if attempt < retries:
            time.sleep(min(2 ** attempt, 4))
    return None, last_err or "model_error"


def _best_rank(state) -> int | None:
    ranks = [e["rank"] for e in state.history if e.get("valid") and "rank" in e]
    return min(ranks) if ranks else None


def _play_episode(game, model: str, effort: str, episode: int, seed: int,
                  max_turns: int, events_path: Path, live_path: Path,
                  call_timeout: int, retries: int) -> tuple[dict, int]:
    state = game.reset(seed)
    while not state.done:
        prompt = game.render(state)
        resp, err = _call(model, prompt, effort, call_timeout, retries)
        if resp is None:
            # 호출 실패: 무효 턴으로 기록하고 에피소드 종료
            state.turn += 1
            state.done = True
            state.stop_reason = "model_error"
            state.history.append({"turn": state.turn, "valid": False,
                                  "error": err, "raw": "", "model_error": True})
            best = _best_rank(state)
            event = {"type": "turn", "episode": episode, "turn": state.turn,
                     "valid": False, "error": err, "best_rank": best,
                     "raw": "", "ts": _now()}
        else:
            action = game.parse(resp.text)
            step_event = game.step(state, action)
            best = _best_rank(state)
            event = _turn_event(episode, step_event, best)
        _append_jsonl(events_path, event)
        _write_json_atomic(live_path, _live(
            model, effort, episode, state.turn, max_turns, "running", event, best))

    result = game.result(state)
    episode_end = {
        "type": "episode_end",
        "episode": episode,
        "solved": result["solved"],
        "turns": result["turns"],
        "best_rank": result["best_rank"],
        "score": result["score"],
        "best_rank_curve": result["best_rank_curve"],
        "max_plateau": result["max_plateau"],    # 고착 지표
        "fixation_sim": result["fixation_sim"],
        "target": state.secret,  # target은 여기(episode_end)에만 등장
        "ts": _now(),
    }
    _append_jsonl(events_path, episode_end)
    return episode_end, result["invalid_actions"]


def _run_participant(game, participant: dict, run_dir: Path, seeds: list[int],
                     max_turns: int, call_timeout: int, retries: int) -> dict:
    model, effort, slug = participant["model"], participant["effort"], participant["slug"]
    mdir = _model_dir(run_dir, slug)
    mdir.mkdir(parents=True, exist_ok=True)
    events_path = mdir / "events.jsonl"
    live_path = mdir / "live.json"

    episode_ends: list[dict] = []
    invalid_total = 0
    for i, seed in enumerate(seeds):
        episode = i + 1  # 1-기반(계약: "episode":1)
        ep_end, invalids = _play_episode(
            game, model, effort, episode, seed, max_turns,
            events_path, live_path, call_timeout, retries)
        episode_ends.append(ep_end)
        invalid_total += invalids

    scores = [e["score"] for e in episode_ends]
    turns = [e["turns"] for e in episode_ends]
    solved = [e for e in episode_ends if e["solved"]]
    plateaus = [e["max_plateau"] for e in episode_ends if e.get("max_plateau") is not None]
    fix_sims = [e["fixation_sim"] for e in episode_ends if e.get("fixation_sim") is not None]
    summary = {
        "model": model,     # 순수 모델 id
        "effort": effort,
        "episodes": episode_ends,
        "mean_score": round(statistics.mean(scores), 6) if scores else 0.0,
        "solve_rate": round(len(solved) / len(episode_ends), 4) if episode_ends else 0.0,
        "median_turns": statistics.median(turns) if turns else None,
        "median_max_plateau": statistics.median(plateaus) if plateaus else None,
        "median_fixation_sim": (round(statistics.median(fix_sims), 8)
                                if fix_sims else None),
        "invalid_actions": invalid_total,
    }
    _write_json_atomic(mdir / "summary.json", summary)

    # 최종 live: phase=done (마지막 턴 상태 유지)
    final = _read_json(live_path)
    final["phase"] = "done"
    final["updated_at"] = _now()
    _write_json_atomic(live_path, final)
    return summary


def run_arena(game_name: str, participants, *, episodes: int = 3,
              max_turns: int | None = None, effort: str = "low",
              seed_base: int | None = None, call_timeout: int = 180,
              run_root: Path | None = None, workers: int | None = None,
              retries: int = 1) -> Path:
    """참가자(모델×effort)들을 동시에 플레이시키고 결과를 slug별로 저장한다.

    participants는 [{"model","effort"}] 또는 "model@effort"/"model" 문자열 리스트를
    수용한다. effort 없는 항목은 default `effort`를 쓴다. 같은 (model,effort)는 중복 제거.
    에피소드 i의 seed = seed_base + i, 전 참가자 공유(동일 에피소드 = 동일 정답).
    동시 실행 상한은 기본 min(len(participants), 12), workers로 조절 가능.
    """
    parts = _normalize_participants(participants, effort)
    if seed_base is None:
        seed_base = int(time.time())
    game = build_game(game_name, max_turns=max_turns)
    max_turns = game.max_turns
    seeds = [seed_base + i for i in range(episodes)]

    run_id = _stamp()
    arena_root = Path(run_root) if run_root is not None else (config.RESULTS / "arena")
    run_dir = arena_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "game": game_name,
        "participants": parts,                    # [{"model","effort","slug"}]
        "models": [p["slug"] for p in parts],     # index 표시용 slug 리스트
        "episodes": episodes,
        "max_turns": max_turns,
        "effort": effort,                         # 기본 effort
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "game_version": game.version,
        "oracle": game.metadata,                  # semantle 오라클 metadata
        "seeds": seeds,                           # 에피소드 e(1-기반) → seeds[e-1]
        "verify": None,
    }
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _index_upsert(arena_root, _index_entry(manifest))

    if workers is None:
        n_workers = min(len(parts), MAX_WORKERS_DEFAULT)
    else:
        n_workers = workers
    n_workers = max(1, n_workers)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_participant, game, p, run_dir, seeds,
                                   max_turns, call_timeout, retries): p
                       for p in parts}
            for fut in as_completed(futures):
                fut.result()  # 예외 전파
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = _now()
        manifest["failure"] = str(exc)
        _write_json_atomic(run_dir / "manifest.json", manifest)
        _index_upsert(arena_root, _index_entry(manifest))
        raise

    manifest["status"] = "done"
    manifest["finished_at"] = _now()
    manifest["verify"] = verify_run(run_dir)
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _index_upsert(arena_root, _index_entry(manifest))
    return run_dir


# ----------------------------------------------------------------------
# 검증 — 저장된 raw를 동일 게임/동일 seed로 재파싱·재스텝해 대조
# ----------------------------------------------------------------------
def _resolve_participants(manifest: dict) -> list[tuple[str, str, str]]:
    """(디렉토리명, 모델 id, effort) 목록.

    v2: manifest.participants의 slug를 그대로 쓴다.
    레거시: participants 없음 → 디렉토리=모델 id, effort=manifest.effort로 유도.
    """
    parts = manifest.get("participants")
    if isinstance(parts, list) and parts:
        return [(p["slug"], p["model"], p.get("effort")) for p in parts]
    effort = manifest.get("effort")
    return [(m, m, effort) for m in manifest.get("models", [])]


def verify_run(run_dir: Path) -> dict:
    """저장된 이벤트를 핀 고정 게임으로 재생해 결과 위변조를 탐지한다.

    임베딩 오라클이 필요하므로 ollama가 없으면 {"ok": None, "skipped": "no-ollama"}.
    similarity는 저장된 반올림 값(round 8) 기준으로 동일성 비교한다.
    참가자 slug 디렉토리를 순회한다(레거시 런은 모델 id 디렉토리로 유도).
    """
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    game_name = manifest.get("game")
    if not embed.available():
        return {"ok": None, "skipped": "no-ollama"}

    game = build_game(game_name, max_turns=int(manifest.get("max_turns", 40)))
    # 게임 버전 인지: 규칙이 바뀐 버전으로 옛 런을 재생하면 정직하지 않다 → skip.
    manifest_version = manifest.get("game_version")
    if manifest_version != game.version:
        return {"ok": None, "skipped": "game-version-mismatch",
                "manifest_version": manifest_version, "current_version": game.version}
    seeds = manifest.get("seeds", [])

    ident_errors = []
    if game.metadata != manifest.get("oracle"):
        ident_errors.append("oracle identity mismatch")

    report = {"ok": True, "models": {}}
    for dir_name, model, _effort in _resolve_participants(manifest):
        events = _read_records(_model_dir(run_dir, dir_name) / "events.jsonl")
        by_ep: dict[int, list[dict]] = {}
        order: list[int] = []
        for ev in events:
            ep = ev.get("episode")
            if ep not in by_ep:
                by_ep[ep] = []
                order.append(ep)
            by_ep[ep].append(ev)

        errors = list(ident_errors)
        for ep in order:
            evs = by_ep[ep]
            if not (isinstance(ep, int) and 1 <= ep <= len(seeds)):
                errors.append(f"episode {ep}: missing seed")
                continue
            state = game.reset(int(seeds[ep - 1]))
            for saved in evs:
                if saved.get("type") != "turn":
                    continue
                replayed = game.step(state, game.parse(saved.get("raw", "")))
                for field in ("valid", "guess", "similarity", "rank", "sim_to_prev"):
                    if replayed.get(field) != saved.get(field):
                        errors.append(
                            f"episode {ep} turn {saved.get('turn')}: {field} mismatch")
            end = next((e for e in evs if e.get("type") == "episode_end"), None)
            if end is not None:
                res = game.result(state)
                if state.secret != end.get("target"):
                    errors.append(f"episode {ep}: target mismatch")
                for field in ("solved", "turns", "best_rank", "best_rank_curve",
                              "score", "max_plateau", "fixation_sim"):
                    if res.get(field) != end.get(field):
                        errors.append(f"episode {ep}: {field} mismatch")
        report["models"][dir_name] = {"ok": not errors, "errors": errors}
        if errors:
            report["ok"] = False
    return report
