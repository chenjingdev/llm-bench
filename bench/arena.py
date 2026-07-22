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

import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import client, config, embed
from .games import build_game
from .games.base import GameState


RAW_LIMIT = 600           # live.json 표시용 스니펫 절단 길이(events.jsonl에는 전문 저장)
MAX_WORKERS_DEFAULT = 32  # 동시 실행 상한 기본값 = 런처 참가자 상한. "전 모델 동시 플레이"가
                          # 제품 약속이므로 기본은 전원 동시 시작, 절감은 workers= 인자로.
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
def _live_last_default(index: int):
    """LIVE_LAST_FIELDS의 결측 기본값 — 첫(주 행동) 필드는 "", 나머지는 None.

    첫 필드는 관례상 행동 라벨(guess/move/kind 등, 문자열)이라 빈 문자열이 자연스럽고,
    뒤따르는 부속 값들은 측정값이라 null이 자연스럽다. 이 규칙이 semantle의 기존
    live.json 값(last_guess="" · last_similarity/last_rank=null)을 그대로 재현한다.
    """
    return "" if index == 0 else None


def _turn_event(game, state, episode: int, step_event: dict) -> dict:
    """게임 step 이벤트 → events.jsonl 턴 레코드(계약 v1 §2).

    valid 턴은 game.TURN_FIELDS 중 step 이벤트에 존재하는 것을, 무효 턴은
    game.INVALID_KEEP 중 존재하는 것 + error를 싣는다(중복 추측의 guess 등 재생
    대조용 필드 보존). 이어서 game.progress(state)를 최상위 병합하고, raw 전문과
    ts를 붙인다. 정답(target)은 어느 턴 이벤트에도 넣지 않는다.
    """
    ev = {"type": "turn", "episode": episode, "turn": step_event["turn"],
          "valid": bool(step_event.get("valid"))}
    if step_event.get("valid"):
        for f in game.TURN_FIELDS:
            if f in step_event:
                ev[f] = step_event[f]
    else:
        for f in game.INVALID_KEEP:
            if f in step_event:  # 예: 중복 추측 무효 턴은 guess를 보존
                ev[f] = step_event[f]
        ev["error"] = step_event.get("error", "invalid")
    ev.update(game.progress(state))
    # 전문 보존: 재생 검증이 이 raw를 다시 parse하므로 절단하면 오탐이 난다.
    ev["raw"] = step_event.get("raw") or ""
    ev["ts"] = _now()
    return ev


def _live(game, state, model: str, effort: str, episode: int, turn: int,
          max_turns: int, phase: str, event: dict | None) -> dict:
    """live.json 조립(계약 v1 §2) — 공통부 + last_<필드> + progress + 스니펫."""
    ev = event or {}
    live = {
        "model": model,     # 순수 모델 id(slug 아님)
        "effort": effort,
        "episode": episode,
        "turn": turn,
        "max_turns": max_turns,
        "phase": phase,
    }
    for i, f in enumerate(game.LIVE_LAST_FIELDS):
        live[f"last_{f}"] = ev.get(f, _live_last_default(i))
    live.update(game.progress(state))
    live["raw_snippet"] = (ev.get("raw", ""))[:RAW_LIMIT]
    live["updated_at"] = _now()
    return live


class _StreamWriter:
    """턴 진행 중 공개 출력을 stream.json으로 노출(관전용 일시 상태).

    stream.json은 재생 검증 대상이 아니다. 쓰기는 throttle초 스로틀하되, 마지막
    상태(finish)는 반드시 기록한다. 재시도로 재호출되면 begin()이 text=""로 리셋한다.
    """

    def __init__(self, path: Path, model: str, effort: str, episode: int,
                 turn: int, throttle: float = 0.2):
        self.path = Path(path)
        self.base = {"model": model, "effort": effort,
                     "episode": episode, "turn": turn}
        self.throttle = throttle
        self.text = ""
        self._last = 0.0

    def _flush(self, done: bool) -> None:
        data = dict(self.base)
        data["text"] = self.text
        data["done"] = done
        data["updated_at"] = _now()
        _write_json_atomic(self.path, data)

    def begin(self) -> None:
        self.text = ""
        self._last = time.monotonic()
        self._flush(False)

    def update(self, so_far: str) -> None:
        self.text = so_far
        now = time.monotonic()
        if now - self._last >= self.throttle:
            self._last = now
            self._flush(False)   # 스로틀로 건너뛴 마지막 상태는 finish가 보장

    def finish(self, text: str | None = None) -> None:
        if text is not None:
            self.text = text
        self._flush(True)


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def _call(model: str, prompt: str, effort: str, timeout: int, retries: int,
          writer: "_StreamWriter"):
    """모델 1턴 호출(재시도 retries회). 실패 시 (None, error).

    호출마다 writer.begin()으로 stream을 리셋(재시도면 text=""부터 다시)하고,
    스트리밍 콜백(writer.update)을 client에 넘긴다.
    """
    last_err = ""
    for attempt in range(retries + 1):
        writer.begin()
        resp = client.call(model, prompt, effort=effort, timeout=timeout,
                           on_text=writer.update)
        if resp.ok:
            return resp, ""
        last_err = resp.error or "model_error"
        if attempt < retries:
            time.sleep(min(2 ** attempt, 4))
    return None, last_err or "model_error"


_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                 "cache_read_input_tokens", "cost_usd", "duration_ms")


def _zero_usage() -> dict:
    """참가자 usage 합계 누산기 초기값."""
    return {"input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cost_usd": 0.0, "duration_ms": 0}


def _accumulate_usage(total: dict, usage: dict) -> None:
    """턴 usage를 참가자 누산기에 제자리 합산(중간 실패해도 그때까지 합계가 남는다)."""
    for k in _USAGE_FIELDS:
        total[k] += usage.get(k, 0)


def _usage_of(resp) -> dict:
    """CallResult → 턴 이벤트 usage 오브젝트(실패/None이면 0)."""
    if resp is None:
        return _zero_usage()
    return {
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cache_creation_input_tokens": resp.cache_creation_input_tokens,
        "cache_read_input_tokens": resp.cache_read_input_tokens,
        "cost_usd": resp.cost_usd,
        "duration_ms": resp.duration_ms,
    }


def _play_episode(game, model: str, effort: str, episode: int, seed: int,
                  max_turns: int, events_path: Path, live_path: Path,
                  stream_path: Path, call_timeout: int, retries: int,
                  usage_total: dict) -> tuple[dict, int]:
    state = game.reset(seed)
    # reset이 모델 호출 없이 미리 채운 자동 오프닝 턴(있으면)을 먼저 이벤트로 방출한다
    # (semantle 1.8.0: 무작위 오프닝 1턴. 다른 게임은 빈 history → 이 루프 무영향).
    # usage=0(추가 토큰 없음). verify는 reset 재도출본과 대조하므로 raw 재생 대상 아님.
    for step_event in list(state.history):
        event = _turn_event(game, state, episode, step_event)
        event["usage"] = _zero_usage()
        _append_jsonl(events_path, event)
        _write_json_atomic(live_path, _live(
            game, state, model, effort, episode, state.turn, max_turns, "running", event))
    while not state.done:
        # 진행 중인 턴 번호(= 다음에 채워질 턴)로 stream sink를 연다.
        writer = _StreamWriter(stream_path, model, effort, episode, state.turn + 1)
        prompt = game.render(state)
        resp, err = _call(model, prompt, effort, call_timeout, retries, writer)
        if resp is None:
            # 호출 실패: 무효 턴으로 기록하고 에피소드 종료. 게임 step을 거치지 않으므로
            # 합성 step 이벤트를 만들어 동일한 턴 레코드 조립 경로(_turn_event)를 탄다.
            state.turn += 1
            state.done = True
            state.stop_reason = "model_error"
            step_event = {"turn": state.turn, "valid": False, "error": err,
                          "raw": "", "model_error": True}
            state.history.append(step_event)
            event = _turn_event(game, state, episode, step_event)
            writer.finish()             # 실패 턴도 done=true로 마감
        else:
            action = game.parse(resp.text)
            step_event = game.step(state, action)
            event = _turn_event(game, state, episode, step_event)
            writer.finish(resp.text)    # 전문 + done=true
        # 사용량 기록(추가만 — raw 전문 보존 계약 무접촉). resp None이면 0.
        usage = _usage_of(resp)
        event["usage"] = usage
        _append_jsonl(events_path, event)
        _accumulate_usage(usage_total, usage)   # 참가자 summary.usage 합계용
        _write_json_atomic(live_path, _live(
            game, state, model, effort, episode, state.turn, max_turns, "running", event))

    result = game.result(state)
    # episode_end(계약 v1 §2): target(state.secret)은 여기에만. 이어서 RESULT_FIELDS를
    # result()에서 복사한다(minefield.mines 등 부가 정답류도 게임이 RESULT_FIELDS로 노출).
    episode_end = {"type": "episode_end", "episode": episode, "target": state.secret}
    for f in game.RESULT_FIELDS:
        episode_end[f] = result[f]
    # 종료 사유(추가만, RESULT_FIELDS 밖 → 재생 대조 무영향). model_error(quota/호출
    # 실패)로 조기 종료된 에피소드를 재사용 판정·웹 표시에서 구분하기 위함.
    episode_end["stop_reason"] = state.stop_reason
    # 에피소드 인스턴스 태그(추가만, RESULT_FIELDS 밖 → 재생 대조 무영향). 레인마다 다른
    # 것은 의도(프롬프트 비동일화로 공급자 배치 상관 제거). verify_run은 프롬프트를
    # 재구성하지 않고 저장 raw를 재파싱·재스텝만 하므로 태그를 재현할 필요가 없다 —
    # 따라서 턴 이벤트가 아니라 episode 단위 이벤트(episode_end)에만 감사용으로 남긴다.
    episode_end["nonce"] = state.nonce
    episode_end["ts"] = _now()
    _append_jsonl(events_path, episode_end)
    # 무효 액션 수는 엔진이 직접 센다(게임 result에 의존하지 않음 — 전 게임 공통).
    invalids = sum(1 for e in state.history if not e.get("valid"))
    return episode_end, invalids


def _exc_str(exc: BaseException) -> str:
    """예외 요약 문자열(타입+메시지, 300자). 정답 누출 방지를 위해 상태는 넣지 않는다."""
    return f"{type(exc).__name__}: {exc}"[:300]


def _build_summary(game, model: str, effort: str, episode_ends: list[dict],
                   invalid_total: int, error: str, usage_total: dict) -> dict:
    """summary.json 조립(계약 v1 §2): 공통부 + game.summary_stats(episode_ends).

    공통부는 전 게임이 RESULT_FIELDS로 노출하는 solved/turns/score에서만 유도한다.
    게임별 집계(semantle의 고착 지표 중앙값 등)는 summary_stats가 median_turns와
    invalid_actions 사이에 병합된다(semantle 기존 키 순서·값 보존). usage는 참가자의
    전 턴 이벤트 usage 합계(추가만) — 웹 승자 산정 동률 처리(턴→시간→토큰)에 쓰인다.
    """
    scores = [e["score"] for e in episode_ends]
    turns = [e["turns"] for e in episode_ends]
    solved = [e for e in episode_ends if e["solved"]]
    summary = {
        "model": model,     # 순수 모델 id
        "effort": effort,
        "episodes": episode_ends,
        "mean_score": round(statistics.mean(scores), 6) if scores else 0.0,
        "solve_rate": round(len(solved) / len(episode_ends), 4) if episode_ends else 0.0,
        "median_turns": statistics.median(turns) if turns else None,
    }
    summary.update(game.summary_stats(episode_ends))
    summary["invalid_actions"] = invalid_total
    usage = dict(usage_total)
    usage["cost_usd"] = round(usage["cost_usd"], 6)   # 부동소수 오차 방지
    summary["usage"] = usage
    if error:
        summary["status"] = "failed"   # 부분 결과 + 실패 마킹(추가만)
        summary["error"] = error
    return summary


def _finalize_live(live_path: Path, game, model: str, effort: str, max_turns: int,
                   error: str) -> None:
    """스레드 종료 시 live.json 마감 — running으로 남지 않게(성공 done / 실패 failed).

    phase="failed"는 계약에 값 추가(웹은 'running'만 특수 취급, 그 외는 done으로 수렴 →
    안전). 실패 시 error 필드도 추가(둘 다 추가만, 기존 필드 제거/개명 없음).
    한 턴도 못 돌았으면 최소 골격을 게임의 LIVE_LAST_FIELDS/progress로 조립한다
    (빈 상태 progress → 결측값). semantle에선 기존 골격과 키·값 동일.
    """
    live = _read_json(live_path)
    if not live:  # 한 턴도 못 돌았으면 최소 골격이라도
        empty = GameState(game.id, game.version, 0, max_turns, "")
        live = {"model": model, "effort": effort, "episode": 0, "turn": 0,
                "max_turns": max_turns}
        for i, f in enumerate(game.LIVE_LAST_FIELDS):
            live[f"last_{f}"] = _live_last_default(i)
        live.update(game.progress(empty))
        live["raw_snippet"] = ""
    live["phase"] = "failed" if error else "done"
    if error:
        live["error"] = error
    live["updated_at"] = _now()
    _write_json_atomic(live_path, live)


def _run_participant(game, participant: dict, run_dir: Path, seeds: list[int],
                     max_turns: int, call_timeout: int, retries: int) -> dict:
    """참가자 하나를 플레이. 예외를 잡아 이 참가자만 실패 처리(런은 계속).

    반환: {"ok": bool, "error": str} — run_arena가 부분 실패를 집계한다.
    """
    model, effort, slug = participant["model"], participant["effort"], participant["slug"]
    mdir = _model_dir(run_dir, slug)
    events_path = mdir / "events.jsonl"
    live_path = mdir / "live.json"
    stream_path = mdir / "stream.json"

    episode_ends: list[dict] = []
    invalid_total = 0
    usage_total = _zero_usage()   # 제자리 합산 — 중간 실패해도 그때까지 합계가 남는다
    error = ""
    try:
        mdir.mkdir(parents=True, exist_ok=True)
        for i, seed in enumerate(seeds):
            episode = i + 1  # 1-기반(계약: "episode":1)
            ep_end, invalids = _play_episode(
                game, model, effort, episode, seed, max_turns,
                events_path, live_path, stream_path, call_timeout, retries, usage_total)
            episode_ends.append(ep_end)
            invalid_total += invalids
    except Exception as exc:
        # 참가자 격리: 이 참가자만 실패. KeyboardInterrupt 등 BaseException은
        # 여기서 잡지 않고 run_arena 상위로 올려 런 전체를 중단시킨다.
        error = _exc_str(exc)
        print(f"[arena] participant {slug} failed: {error}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # 어떤 경로로 끝나든 summary/live를 마감(live가 running으로 남지 않게).
        summary = _build_summary(game, model, effort, episode_ends, invalid_total,
                                 error, usage_total)
        _write_json_atomic(mdir / "summary.json", summary)
        _finalize_live(live_path, game, model, effort, max_turns, error)
    return {"ok": not error, "error": error}


# ----------------------------------------------------------------------
# 결과 재사용 — 측정 경제(계약 §9): 동일 측정 조건의 완주 결과를 새 런에 편입
# ----------------------------------------------------------------------
def measurement_key(manifest_like: dict) -> str:
    """측정 조건 지문 — game/game_version/oracle(metadata 전체)/seeds/max_turns의 sha256.

    episodes는 seeds 길이에 내포된다. 오라클 교체·게임 버전 범프는 oracle/game_version을
    통해 키를 바꿔 옛 결과를 자동 무효화한다. manifest.json은 이 다섯 필드를 모두
    담으므로 현재 런과 과거 런에 동일하게 적용해 키를 비교할 수 있다.
    """
    payload = {
        "game": manifest_like.get("game"),
        "game_version": manifest_like.get("game_version"),
        "oracle": manifest_like.get("oracle"),
        "seeds": manifest_like.get("seeds"),
        "max_turns": manifest_like.get("max_turns"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _participant_complete(pdir: Path, episodes: int) -> bool:
    """재사용 가능한 완주 참가자인가 — summary에 실패 마킹 없고 episode_end 수==episodes,
    그리고 어떤 에피소드도 model_error로 끝나지 않아야 한다.

    quota/호출 실패로 1턴 만에 끝난 에피소드가 그 모델의 영구 결과로 재사용되면 보드가
    조용히 오염된다 → 배제. 단 stop_reason 부재(구형 런)는 하위 호환으로 제외하지 않는다.
    """
    pdir = Path(pdir)
    summary = _read_json(pdir / "summary.json")
    if not summary or "status" in summary:   # status는 실패 참가자에만 추가됨
        return False
    ends = [e for e in _read_records(pdir / "events.jsonl")
            if e.get("type") == "episode_end"]
    if len(ends) != episodes:
        return False
    return not any(e.get("stop_reason") == "model_error" for e in ends)


def _copy_participant(src_dir: Path, dst_dir: Path) -> None:
    """참가자 결과 디렉토리 복사 — stream.json(일시 관전 상태)은 제외한다."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(src_dir).iterdir():
        if item.name == "stream.json" or not item.is_file():
            continue
        shutil.copy2(item, dst_dir / item.name)


def _apply_reuse(arena_root: Path, key: str, parts: list[dict], run_dir: Path,
                 episodes: int, cur_run_id: str) -> None:
    """index를 최신순 스캔해 조건 일치·완주·미실패 참가자 디렉토리를 새 런에 복사.

    복사한 참가자는 parts[i]에 reused_from="<원본 run_id>"를 기록(제자리 변경)해
    run_arena가 스레드를 만들지 않게 한다. 재사용 소스는 과거 런만(cur_run_id 제외).
    """
    pending = {p["slug"]: p for p in parts if "reused_from" not in p}
    if not pending:
        return
    index = _read_json(arena_root / "index.json")
    runs = index.get("runs")
    runs = runs if isinstance(runs, list) else []
    for entry in runs:                          # index는 최신이 앞
        if not pending:
            break
        src_run_id = entry.get("run_id")
        if not src_run_id or src_run_id == cur_run_id:
            continue
        src_dir = Path(arena_root) / src_run_id
        src_manifest = _read_json(src_dir / "manifest.json")
        if not src_manifest or measurement_key(src_manifest) != key:
            continue
        failed_slugs = {f.get("slug")
                        for f in src_manifest.get("failed_participants", [])}
        for slug in list(pending):
            if slug in failed_slugs:            # 실패 참가자 결과는 재사용 금지
                continue
            src_pdir = src_dir / "models" / slug
            if not _participant_complete(src_pdir, episodes):
                continue
            _copy_participant(src_pdir, run_dir / "models" / slug)
            pending[slug]["reused_from"] = src_run_id
            del pending[slug]


def run_arena(game_name: str, participants, *, episodes: int = 3,
              max_turns: int | None = None, effort: str = "low",
              seed_base: int | None = None, call_timeout: int = 180,
              run_root: Path | None = None, workers: int | None = None,
              retries: int = 1, reuse: bool = True,
              repeat_seed: bool = False) -> Path:
    """참가자(모델×effort)들을 동시에 플레이시키고 결과를 slug별로 저장한다.

    participants는 [{"model","effort"}] 또는 "model@effort"/"model" 문자열 리스트를
    수용한다. effort 없는 항목은 default `effort`를 쓴다. 같은 (model,effort)는 중복 제거.
    에피소드 i의 seed = seed_base + i(서로 다른 문제), 전 참가자 공유(동일 에피소드 =
    동일 정답). repeat_seed=True면 seeds = [seed_base]*episodes — 같은 문제를 N회 풀어
    안정성(표본 편차)을 잰다. 동시 실행 상한은 기본 min(실측 참가자수, 32).

    reuse=True(기본)이면 동일 측정 조건(measurement_key)의 완주 참가자 결과를 과거 런에서
    복사해 재측정을 생략한다("새 모델만 실측"). --no-reuse/reuse=False면 전부 재측정.
    measurement_key는 seeds 리스트를 포함하므로 [S,S](반복)와 [S,S+1](순차)·[S](단판)이
    자동으로 다른 키가 되어 재사용이 섞이지 않는다.
    """
    parts = _normalize_participants(participants, effort)
    if seed_base is None:
        seed_base = int(time.time())
    # build 전에 확정. repeat_seed면 전판 동일 문제([S]*N), 아니면 서로 다른 문제([S+i]).
    seeds = ([seed_base] * episodes if repeat_seed
             else [seed_base + i for i in range(episodes)])

    run_id = _stamp()
    arena_root = Path(run_root) if run_root is not None else (config.RESULTS / "arena")
    run_dir = arena_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 준비 단계 가시화(계약 부록 B): ko-semantle/ko-minefield의 build_game은 콜드 스타트
    # 시 오라클 임베딩 로딩으로 수십 초 걸린다. 그 전에 예비 manifest(status:"preparing")를
    # 남겨 런이 index에서 사라지지 않게 한다(웹 런처가 폴링 중 조용히 포기하는 것 방지).
    # game_version/oracle/measurement_key/verify는 아직 모르므로 생략 — 최종 manifest가
    # 오늘과 동일한 필드셋으로 덮어쓴다. max_turns는 요청값 그대로(미지정이면 null).
    started_at = _now()   # 런 시작 시각은 준비 단계 포함 — 최종 manifest에도 이 값
    manifest = {
        "run_id": run_id,
        "game": game_name,
        "participants": parts,
        "models": [p["slug"] for p in parts],
        "episodes": episodes,
        "max_turns": max_turns,
        "effort": effort,
        "status": "preparing",
        "started_at": started_at,
        "finished_at": None,
        "pid": os.getpid(),   # 웹 정지 기능이 이 pid를 killpg 대상으로 읽는다(준비 단계도 포함)
        "seeds": seeds,
    }
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _index_upsert(arena_root, _index_entry(manifest))

    # 빌드 실패 정직성: ollama 다운 등으로 실패하면 "preparing"으로 영원히 남지 않게
    # failed로 마감하고 예외를 올린다.
    try:
        game = build_game(game_name, max_turns=max_turns)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = _exc_str(exc)
        manifest["finished_at"] = _now()
        _write_json_atomic(run_dir / "manifest.json", manifest)
        _index_upsert(arena_root, _index_entry(manifest))
        raise
    max_turns = game.max_turns

    manifest = {
        "run_id": run_id,
        "game": game_name,
        "participants": parts,                    # [{"model","effort","slug"}]
        "models": [p["slug"] for p in parts],     # index 표시용 slug 리스트
        "episodes": episodes,
        "max_turns": max_turns,
        "effort": effort,                         # 기본 effort
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "pid": os.getpid(),                       # 웹 정지 기능의 killpg 대상(추가 필드)
        "game_version": game.version,
        "oracle": game.metadata,                  # semantle 오라클 metadata
        "seeds": seeds,                           # 에피소드 e(1-기반) → seeds[e-1]
        "verify": None,
    }
    # 측정 조건 키(추가 필드) + 재사용 편입. _apply_reuse는 parts를 제자리 변경해
    # reused_from을 기록하므로 manifest.participants에 그대로 반영된다(추가만).
    manifest["measurement_key"] = measurement_key(manifest)
    if reuse:
        _apply_reuse(arena_root, manifest["measurement_key"], parts, run_dir,
                     episodes, run_id)
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _index_upsert(arena_root, _index_entry(manifest))

    to_run = [p for p in parts if "reused_from" not in p]   # 재사용분은 스레드 생략
    if workers is None:
        n_workers = min(len(to_run), MAX_WORKERS_DEFAULT)
    else:
        n_workers = workers
    n_workers = max(1, n_workers)

    failed: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_participant, game, p, run_dir, seeds,
                                   max_turns, call_timeout, retries): p
                       for p in to_run}
            for fut in as_completed(futures):
                p = futures[fut]
                res = fut.result()  # _run_participant는 Exception을 삼키고 상태를 반환
                if not res.get("ok", True):
                    failed.append({"slug": p["slug"], "model": p["model"],
                                   "effort": p["effort"], "error": res.get("error", "")})
    except BaseException as exc:
        # 참가자 단위로 못 잡은 치명 오류(KeyboardInterrupt 등)만 여기 도달 → 런 전체 실패.
        manifest["status"] = "failed"
        manifest["finished_at"] = _now()
        manifest["failure"] = _exc_str(exc)
        _write_json_atomic(run_dir / "manifest.json", manifest)
        _index_upsert(arena_root, _index_entry(manifest))
        raise

    # 일부 참가자가 실패해도 완주분은 살린다: status="done" + failed_participants 명시.
    manifest["status"] = "done"
    manifest["finished_at"] = _now()
    if failed:
        manifest["failed_participants"] = failed
    manifest["verify"] = verify_run(run_dir)  # 완주 참가자만 실질 검증(실패분은 이벤트 없음)
    _write_json_atomic(run_dir / "manifest.json", manifest)
    _index_upsert(arena_root, _index_entry(manifest))
    return run_dir


# ----------------------------------------------------------------------
# 검증 — 저장된 raw를 동일 게임/동일 seed로 재파싱·재스텝해 대조
# ----------------------------------------------------------------------
def _within_tol(a, b, tol: float) -> bool:
    """절대 오차 tol 이내면 일치. 숫자 리스트는 같은 길이 + 원소별 오차 이내.

    None vs None은 일치, None vs 숫자는 불일치(계기 노이즈가 값을 사라지게 만들지는
    않으므로). bool은 정확 비교(True가 1과 뭉개지지 않게).
    """
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return False
        return all(_within_tol(x, y, tol) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


def _fields_match(field: str, replayed, saved, tol_map: dict) -> bool:
    """재생값 vs 저장값 동일성 판정. TOLERANT_FIELDS면 오차 비교, 아니면 정확 비교."""
    tol = tol_map.get(field)
    if tol is None:
        return replayed == saved
    return _within_tol(replayed, saved, tol)


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

    임베딩 오라클이 필요한 게임(game.needs_ollama)만 ollama 게이트를 적용한다 —
    없으면 {"ok": None, "skipped": "no-ollama"}. 비오라클 게임은 ollama 없이 검증한다.
    turn 대조 필드는 game.TURN_FIELDS(valid 턴에만 복사되므로 valid 턴에서만 대조),
    episode_end 대조 필드는 game.RESULT_FIELDS를 쓴다(하드코딩 없음). "valid"는 항상
    정확 비교. game.TOLERANT_FIELDS에 있는 필드는 절대 오차 이내를 일치로 본다(임베딩
    오라클의 동시 요청 노이즈 수용 — 근거는 base.py 프로토콜 주석).
    참가자 slug 디렉토리를 순회한다(레거시 런은 모델 id 디렉토리로 유도).
    """
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    game_name = manifest.get("game")
    try:
        game = build_game(game_name, max_turns=int(manifest.get("max_turns", 40)))
    except Exception:
        # 게임 구성 실패(예: 오라클 필요 게임인데 ollama 부재) → 재생 불가로 판단해 skip.
        # ollama가 살아 있는데도 실패했다면 진짜 오류이므로 그대로 올린다.
        if not embed.available():
            return {"ok": None, "skipped": "no-ollama"}
        raise
    # 임베딩 오라클이 필요한 게임만 ollama 게이트를 적용(비오라클 게임은 무관하게 진행).
    if game.needs_ollama and not embed.available():
        return {"ok": None, "skipped": "no-ollama"}
    # 게임 버전 인지: 규칙이 바뀐 버전으로 옛 런을 재생하면 정직하지 않다 → skip.
    manifest_version = manifest.get("game_version")
    if manifest_version != game.version:
        return {"ok": None, "skipped": "game-version-mismatch",
                "manifest_version": manifest_version, "current_version": game.version}
    seeds = manifest.get("seeds", [])

    tol_map = getattr(game, "TOLERANT_FIELDS", {})   # 미선언 게임은 전 필드 정확 비교
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
            # 저장된 nonce를 reset에 넘겨 nonce 파생 결과를 동일하게 재도출할 수 있게 한다
            # (현재 semantle은 nonce 무의존이지만, 향후 게임/재사용 대비해 파이프라인 유지).
            # nonce 인자를 안 받는 게임(구형/스텁 reset)은 없이 재생(재생 무해).
            end = next((e for e in evs if e.get("type") == "episode_end"), None)
            seed_val = int(seeds[ep - 1])
            try:
                state = game.reset(seed_val, nonce=(end.get("nonce") if end else None))
            except TypeError:
                state = game.reset(seed_val)
            # reset이 미리 채운 자동 오프닝 턴(앞 n_auto개)은 step 재생이 아니라 재도출본
            # (state.history)과 대조한다 — raw=""를 재파싱하면 안 되기 때문. 나머지 모델 턴만
            # step 재생. 오프닝 없는 게임은 n_auto=0이라 전부 step 재생(기존 동작 그대로).
            saved_turns = [e for e in evs if e.get("type") == "turn"]
            n_auto = len(state.history)
            for i, saved in enumerate(saved_turns):
                computed = (state.history[i] if i < n_auto
                            else game.step(state, game.parse(saved.get("raw", ""))))
                # "valid"는 항상 정확 비교. TURN_FIELDS는 valid 턴에만 저장되므로(무효
                # 턴 레코드의 progress 파생 필드를 재생 step 이벤트와 오대조하지 않게)
                # saved["valid"]가 참인 필드만 대조한다.
                for field in ("valid", *game.TURN_FIELDS):
                    if field != "valid" and not saved.get("valid"):
                        continue
                    if not _fields_match(field, computed.get(field),
                                         saved.get(field), tol_map):
                        errors.append(
                            f"episode {ep} turn {saved.get('turn')}: {field} mismatch")
            if end is not None:                       # (nonce 재도출용으로 위에서 이미 조회)
                res = game.result(state)
                if state.secret != end.get("target"):
                    errors.append(f"episode {ep}: target mismatch")
                for field in game.RESULT_FIELDS:
                    if not _fields_match(field, res.get(field), end.get(field), tol_map):
                        errors.append(f"episode {ep}: {field} mismatch")
        report["models"][dir_name] = {"ok": not errors, "errors": errors}
        if errors:
            report["ok"] = False
    return report
