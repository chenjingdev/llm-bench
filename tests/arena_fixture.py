"""Mindmatch 관전 콘솔 픽스처 (계약 v2 — 참가자 = 모델×effort).

화면의 모든 상태 + 두 저장 형태가 보이도록 results/arena 트리를 합성한다:
  - 해결 성공(1위) + 정답 공개(끝난 에피소드)
  - 미해결(다양한 best_rank: 선두 hot ~ 후미 laggard)
  - 무효 턴 섞임(형식 오류 = guess 없음, 중복 추측 = guess 있음)
  - 진행 중 참가자(LIVE 런 ep2, episode_end 없음 → 정답 은닉)
  - 완료 참가자(끝난 에피소드)
  - **같은 모델 다른 effort** 참가자(h4.5@low vs h4.5@high) — 레인 두 개 (v2 핵심)
  - v2 형태(manifest.participants, dir=slug, live/summary에 effort)
  - 레거시 형태(participants 없음, dir=모델 id, effort 없음) — 둘 다 렌더 확인

test와 스크린샷 서버가 함께 쓴다:
  from tests.arena_fixture import build; build(tmp_path)
  python3 -m tests.arena_fixture <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REF_WORDS = 178
_T0 = 1_700_000_000


def _ts(step: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(_T0 + step * 17).isoformat()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def _write_lines(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _write_stream(run_dir: Path, slug: str, obj) -> None:
    _write(run_dir / "models" / slug / "stream.json", obj)


def _usage(dur_ms: int, out_tok: int, in_tok: int | None = None) -> dict:
    """엔진이 턴 이벤트·summary에 싣는 usage 오브젝트(동일 키 세트)."""
    it = in_tok if in_tok is not None else out_tok * 4
    return {
        "input_tokens": it, "output_tokens": out_tok,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "cost_usd": round(it * 3e-6 + out_tok * 1.5e-5, 6),
        "duration_ms": dur_ms,
    }


# --- 턴 스펙 DSL --------------------------------------------------------
# ("v", 단어, 유사도, 순위)       유효 추측
# ("dup", 단어)                    무효: 중복 추측(guess 있음)
# ("fmt", 원문)                    무효: 형식 오류(guess 없음)
def _build_episode(events, episode, specs, *, finished, target, step_ref, usage_pt=None):
    best = None
    curve = []
    last_valid = None
    invalid = 0
    turn = 0
    for spec in specs:
        turn += 1
        step_ref[0] += 1
        kind = spec[0]
        if kind == "v":
            _, word, sim, rank = spec
            if best is None or rank < best:
                best = rank
            last_valid = {"guess": word, "similarity": sim, "rank": rank}
            events.append({
                "type": "turn", "episode": episode, "turn": turn, "valid": True,
                "guess": word, "similarity": round(sim, 8), "rank": rank,
                "best_rank": best, "raw": f"GUESS {word}", "ts": _ts(step_ref[0]),
            })
        elif kind == "dup":
            _, word = spec
            invalid += 1
            events.append({
                "type": "turn", "episode": episode, "turn": turn, "valid": False,
                "guess": word, "error": "이미 시도한 단어입니다(중복 추측)",
                "best_rank": best, "raw": f"GUESS {word}", "ts": _ts(step_ref[0]),
            })
        elif kind == "fmt":
            _, raw = spec
            invalid += 1
            events.append({
                "type": "turn", "episode": episode, "turn": turn, "valid": False,
                "error": "exactly one GUESS <한국어 단어> line required",
                "best_rank": best, "raw": raw, "ts": _ts(step_ref[0]),
            })
        else:
            raise ValueError(kind)
        if usage_pt is not None:      # 턴마다 usage(구형 런은 usage_pt=None → 키 없음)
            events[-1]["usage"] = _usage(usage_pt[0], usage_pt[1])
        curve.append(best if best is not None else REF_WORDS)

    solved = bool(best == 1)
    result = {"best_rank": best, "turns": turn, "solved": solved, "curve": curve,
              "last_valid": last_valid, "invalid": invalid, "target": target}
    if finished:
        step_ref[0] += 1
        score = round(max(0.0, 1 - (0 if best is None else (best - 1)) / (REF_WORDS - 1)) * 0.9, 6)
        events.append({
            "type": "episode_end", "episode": episode, "solved": solved,
            "turns": turn, "best_rank": best, "score": score,
            "best_rank_curve": curve, "target": target, "ts": _ts(step_ref[0]),
        })
        result["score"] = score
    return result


def _write_participant(run_dir, slug, model, effort, episodes, *,
                       current_ep, phase, max_turns, legacy=False, usage_pt=None):
    """참가자 하나(디렉토리=slug)를 쓴다.

    legacy=True면 live/summary에 effort를 넣지 않는다(레거시 파생 경로 테스트).
    usage_pt=(dur_ms, out_tok)면 턴마다 usage를 싣고 summary["usage"]에 합계를 쓴다.
    None이면 usage 없는 구형 참가자(정렬 꼬리 결측 케이스).
    """
    events = []
    step = [hash(slug) % 1000]
    ep_results = {}
    finished_results = []
    for ep in sorted(episodes):
        specs, finished, target = episodes[ep]
        r = _build_episode(events, ep, specs, finished=finished, target=target,
                           step_ref=step, usage_pt=usage_pt)
        ep_results[ep] = r
        if finished:
            finished_results.append((ep, r))

    _write_lines(run_dir / "models" / slug / "events.jsonl", events)

    cur = ep_results[current_ep]
    lv = cur["last_valid"]
    live = {
        "model": model, "episode": current_ep, "turn": cur["turns"],
        "max_turns": max_turns, "phase": phase,
        "last_guess": lv["guess"] if lv else "",
        "last_similarity": round(lv["similarity"], 8) if lv else None,
        "last_rank": lv["rank"] if lv else None,
        "best_rank": cur["best_rank"],
        "raw_snippet": f"GUESS {lv['guess']}" if lv else "",
        "updated_at": _ts(step[0]),
    }
    if not legacy:
        live["effort"] = effort
    _write(run_dir / "models" / slug / "live.json", live)

    eps_out, scores, solves, turns_l = [], [], [], []
    inval = 0
    for ep, r in finished_results:
        eps_out.append({
            "type": "episode_end", "episode": ep, "solved": r["solved"],
            "turns": r["turns"], "best_rank": r["best_rank"], "score": r["score"],
            "best_rank_curve": r["curve"], "target": r["target"], "ts": _ts(step[0]),
        })
        scores.append(r["score"])
        solves.append(1 if r["solved"] else 0)
        turns_l.append(r["turns"])
    for ep in episodes:
        inval += ep_results[ep]["invalid"]
    summary = {
        "model": model, "episodes": eps_out,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "solve_rate": round(sum(solves) / len(solves), 4) if solves else 0.0,
        "median_turns": sorted(turns_l)[len(turns_l) // 2] if turns_l else 0,
        "invalid_actions": inval,
    }
    if not legacy:
        summary["effort"] = effort
    if usage_pt is not None:      # 전 에피소드 턴 합 = per-turn × 총 턴 수
        tot = sum(r["turns"] for r in ep_results.values())
        summary["usage"] = _usage(usage_pt[0] * tot, usage_pt[1] * tot)
    _write(run_dir / "models" / slug / "summary.json", summary)


def _oracle():
    return {"game": "ko-semantle", "version": "1.0.0",
            "type": "exact-cosine-reference-rank",
            "embedding_model": "qwen3-embedding:4b",
            "reference_words": REF_WORDS,
            "rank_scope": "pinned-reference-vocabulary"}


def _manifest_v2(run_id, participants, episodes, max_turns, status, started, finished,
                 failure=None, seeds=None):
    slugs = [p["slug"] for p in participants]
    man = {
        "run_id": run_id, "game": "ko-semantle", "models": slugs,
        "participants": participants,
        "episodes": episodes, "max_turns": max_turns,
        "status": status, "started_at": started, "finished_at": finished,
        "game_version": "1.0.0", "oracle": _oracle(),
        "seeds": seeds if seeds is not None else [1784015657, 42],
        "verify": {"ok": True, "models": {s: {"ok": True, "errors": []} for s in slugs}},
    }
    if failure is not None:          # 중단 런에만 사유를 싣는다(정상 런엔 키 없음).
        man["failure"] = failure
    return man


def _manifest_preparing(run_id, game, participants, episodes, max_turns, effort,
                        started, seeds):
    """예비 manifest(build_game 전 기록): status='preparing'. 오라클/game_version/
    measurement_key/verify는 이 시점 부재(키 자체 없음). models/ 디렉터리도 아직 없다."""
    return {
        "run_id": run_id, "game": game,
        "models": [p["slug"] for p in participants],
        "participants": participants,
        "episodes": episodes, "max_turns": max_turns, "effort": effort,
        "status": "preparing", "started_at": started, "finished_at": None,
        "seeds": seeds,
    }


def _manifest_legacy(run_id, models, episodes, max_turns, effort, status, started, finished):
    return {
        "run_id": run_id, "game": "ko-semantle", "models": models,
        "episodes": episodes, "max_turns": max_turns, "effort": effort,
        "status": status, "started_at": started, "finished_at": finished,
        "game_version": "1.0.0", "oracle": _oracle(), "seeds": [1784015657],
        "verify": {"ok": True, "models": {m: {"ok": True, "errors": []} for m in models}},
    }


# ======================================================================
# 신규 3게임 픽스처 (계약 §3~§5 이벤트 스키마 그대로 = 엔진 산출물의 형태)
# ======================================================================
def _game_oracle(game: str) -> dict:
    if game == "ko-minefield":
        o = _oracle()
        o.update({"game": "ko-minefield", "lives": 3, "mines": 2,
                  "boom_rank": 3, "warn_rank": 15})
        return o
    if game == "ko-maze":
        return {"game": "ko-maze", "version": "1.0.0",
                "type": "deterministic-perfect-maze", "grid": 7,
                "rank_scope": "n/a"}
    return {"game": "ko-rulelab", "version": "1.0.0",
            "type": "deterministic-rule-oracle", "rank_scope": "n/a"}


def _manifest_game(run_id, game, participants, episodes, max_turns, status,
                   started, finished, failure=None):
    slugs = [p["slug"] for p in participants]
    man = {
        "run_id": run_id, "game": game, "models": slugs,
        "participants": participants,
        "episodes": episodes, "max_turns": max_turns,
        "status": status, "started_at": started, "finished_at": finished,
        "game_version": "1.0.0", "oracle": _game_oracle(game), "seeds": [1784015657],
        "verify": {"ok": True, "models": {s: {"ok": True, "errors": []} for s in slugs}},
    }
    if failure is not None:
        man["failure"] = failure
    return man


# --- ko-rulelab -------------------------------------------------------
# specs DSL: ("test", a, b, out) | ("answer", [v1..v5], correct) | ("fmt", raw)
def _rulelab_participant(run_dir, slug, model, effort, specs, *,
                         finished, rule, max_turns, phase, current_ep=1, usage_pt=None):
    events = []
    step = [hash(slug) % 1000]
    experiments = 0
    answered = False
    dup = 0
    seen = set()
    correct = None
    turn = 0
    last_kind = last_input = last_output = None
    for spec in specs:
        turn += 1
        step[0] += 1
        kind = spec[0]
        if kind == "test":
            _, a, b, out = spec
            experiments += 1
            key = (a, b)
            if key in seen:
                dup += 1
            else:
                seen.add(key)
            last_kind, last_input, last_output = "test", [a, b], out
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": True,
                "kind": "test", "input": [a, b], "output": out,
                "experiments": experiments, "answered": answered,
                "raw": f"TEST {a} {b}", "ts": _ts(step[0]),
            })
        elif kind == "answer":
            _, arr, corr = spec
            answered, correct = True, corr
            last_kind, last_input, last_output = "answer", None, None
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": True,
                "kind": "answer", "answer": arr, "correct": corr,
                "experiments": experiments, "answered": True,
                "raw": "ANSWER " + " ".join(map(str, arr)), "ts": _ts(step[0]),
            })
        elif kind == "fmt":
            _, raw = spec
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": False,
                "error": "정확히 한 줄의 TEST 또는 ANSWER가 필요합니다",
                "experiments": experiments, "answered": answered,
                "raw": raw, "ts": _ts(step[0]),
            })
        else:
            raise ValueError(kind)
        if usage_pt is not None:
            events[-1]["usage"] = _usage(usage_pt[0], usage_pt[1])
    solved = bool(correct == 5)
    score = round((correct or 0) / 5, 6)
    if finished:
        step[0] += 1
        events.append({
            "type": "episode_end", "episode": current_ep, "target": rule,
            "solved": solved, "turns": turn, "score": score,
            "experiments": experiments, "duplicate_tests": dup,
            "correct": correct if correct is not None else 0, "ts": _ts(step[0]),
        })
    _write_lines(run_dir / "models" / slug / "events.jsonl", events)
    _turn_evs = [e for e in events if e.get("type") == "turn"]
    live = {
        "model": model, "effort": effort, "episode": current_ep, "turn": turn,
        "max_turns": max_turns, "phase": phase,
        "last_kind": last_kind, "last_input": last_input, "last_output": last_output,
        "experiments": experiments, "answered": answered,
        "raw_snippet": _turn_evs[-1]["raw"] if _turn_evs else "", "updated_at": _ts(step[0]),
    }
    _write(run_dir / "models" / slug / "live.json", live)
    summary = {
        "model": model, "effort": effort, "episodes": [],
        "mean_score": score if finished else 0.0,
        "solve_rate": 1.0 if solved else 0.0, "median_turns": turn,
        "median_experiments": experiments, "answer_rate": 1.0 if answered else 0.0,
    }
    if usage_pt is not None:
        summary["usage"] = _usage(usage_pt[0] * turn, usage_pt[1] * turn)
    if finished:
        summary["episodes"] = [{
            "type": "episode_end", "episode": current_ep, "solved": solved,
            "turns": turn, "score": score, "experiments": experiments,
            "duplicate_tests": dup, "correct": correct if correct is not None else 0,
            "target": rule, "ts": _ts(step[0]),
        }]
    _write(run_dir / "models" / slug / "summary.json", summary)


# --- ko-maze ----------------------------------------------------------
# specs DSL: ("move", dir, [x,y], open_str, bearing, dist, ok) | ("fmt", raw)
def _maze_participant(run_dir, slug, model, effort, specs, *,
                      finished, target, max_turns, phase,
                      start=(0, 0), start_dist=12, shortest=12, current_ep=1, usage_pt=None):
    events = []
    step = [hash(slug) % 1000]
    visited = {tuple(start)}
    bumps = 0
    turn = 0
    moves = 0
    revisits = 0
    last_move = last_pos = last_bearing = None
    dist = start_dist
    dist_curve = []
    for spec in specs:
        turn += 1
        step[0] += 1
        if spec[0] == "move":
            _, d, pos, opn, bearing, dd, ok = spec
            moves += 1
            if ok:
                if tuple(pos) in visited:
                    revisits += 1
                visited.add(tuple(pos))
            else:
                bumps += 1
            dist = dd
            last_move, last_pos, last_bearing = d, pos, bearing
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": True,
                "move": d, "ok": ok, "pos": pos, "open": opn,
                "bearing": bearing, "dist": dd,
                "explored": round(len(visited) / 49, 4), "bumps": bumps,
                "raw": f"MOVE {d}", "ts": _ts(step[0]),
            })
        elif spec[0] == "fmt":
            _, raw = spec
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": False,
                "error": "정확히 한 줄의 MOVE <북|남|동|서>가 필요합니다",
                "dist": dist, "explored": round(len(visited) / 49, 4), "bumps": bumps,
                "raw": raw, "ts": _ts(step[0]),
            })
        else:
            raise ValueError(spec[0])
        if usage_pt is not None:
            events[-1]["usage"] = _usage(usage_pt[0], usage_pt[1])
        dist_curve.append(dist)
    solved = bool(finished and last_pos is not None
                  and target == f"{last_pos[0]},{last_pos[1]}")
    explored_ratio = round(len(visited) / 49, 4)
    min_dist = min(dist_curve) if dist_curve else start_dist
    score = round((max_turns - turn + 1) / max_turns, 6) if solved else 0.0
    if finished:
        step[0] += 1
        events.append({
            "type": "episode_end", "episode": current_ep, "target": target,
            "solved": solved, "turns": turn, "score": score, "moves": moves,
            "bumps": bumps, "revisits": revisits, "explored_ratio": explored_ratio,
            "path_efficiency": round(shortest / moves, 6) if (solved and moves) else None,
            "min_dist": min_dist, "dist_curve": dist_curve, "ts": _ts(step[0]),
        })
    _write_lines(run_dir / "models" / slug / "events.jsonl", events)
    live = {
        "model": model, "effort": effort, "episode": current_ep, "turn": turn,
        "max_turns": max_turns, "phase": phase,
        "last_move": last_move, "last_pos": last_pos, "last_bearing": last_bearing,
        "dist": dist, "explored": explored_ratio, "bumps": bumps,
        "raw_snippet": f"MOVE {last_move}" if last_move else "",
        "updated_at": _ts(step[0]),
    }
    _write(run_dir / "models" / slug / "live.json", live)
    summary = {
        "model": model, "effort": effort, "episodes": [],
        "mean_score": score if finished else 0.0,
        "solve_rate": 1.0 if solved else 0.0, "median_turns": turn,
        "median_min_dist": min_dist, "median_explored": explored_ratio,
    }
    if usage_pt is not None:
        summary["usage"] = _usage(usage_pt[0] * turn, usage_pt[1] * turn)
    if finished:
        summary["episodes"] = [{
            "type": "episode_end", "episode": current_ep, "solved": solved,
            "turns": turn, "score": score, "moves": moves, "bumps": bumps,
            "revisits": revisits, "explored_ratio": explored_ratio,
            "min_dist": min_dist, "dist_curve": dist_curve, "target": target,
            "ts": _ts(step[0]),
        }]
    _write(run_dir / "models" / slug / "summary.json", summary)


# --- ko-minefield -----------------------------------------------------
# specs DSL: ("normal", g, sim, rank, sim_to_prev) | ("warn", g, sim, rank, stp)
#          | ("boom", g, lives_after) | ("win", g) | ("dup", g) | ("fmt", raw)
def _mine_participant(run_dir, slug, model, effort, specs, *,
                      finished, target, mines, max_turns, phase,
                      start_lives=3, current_ep=1, usage_pt=None):
    events = []
    step = [hash(slug) % 1000]
    lives = start_lives
    best = None
    curve = []
    booms = warns = turn = 0
    solved = False
    prev_sim = None
    for spec in specs:
        turn += 1
        step[0] += 1
        kind = spec[0]
        if kind in ("normal", "warn", "win"):
            if kind == "win":
                guess, sim, rank, stp = spec[1], 1.0, 1, prev_sim
                solved = True
            else:
                _, guess, sim, rank, stp = spec
                if kind == "warn":
                    warns += 1
            if best is None or rank < best:
                best = rank
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": True,
                "guess": guess, "similarity": round(sim, 8), "rank": rank,
                "sim_to_prev": stp, "mine_event": ("warn" if kind == "warn" else None),
                "lives": lives, "best_rank": best,
                "raw": f"GUESS {guess}", "ts": _ts(step[0]),
            })
            prev_sim = sim
        elif kind == "boom":
            _, guess, lv = spec
            lives = lv
            booms += 1
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": True,
                "guess": guess, "mine_event": "boom", "lives": lives,
                "best_rank": best, "raw": f"GUESS {guess}", "ts": _ts(step[0]),
            })
            prev_sim = None
        elif kind == "dup":
            _, guess = spec
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": False,
                "guess": guess, "error": "이미 시도한 단어입니다(중복 추측)",
                "lives": lives, "best_rank": best,
                "raw": f"GUESS {guess}", "ts": _ts(step[0]),
            })
        elif kind == "fmt":
            _, raw = spec
            events.append({
                "type": "turn", "episode": current_ep, "turn": turn, "valid": False,
                "error": "exactly one GUESS <한국어 단어> line required",
                "lives": lives, "best_rank": best, "raw": raw, "ts": _ts(step[0]),
            })
        else:
            raise ValueError(kind)
        if usage_pt is not None:
            events[-1]["usage"] = _usage(usage_pt[0], usage_pt[1])
        curve.append(best if best is not None else REF_WORDS)
    mined = bool(finished and not solved and lives == 0)
    stop_reason = "solved" if solved else "mined" if mined else "max_turns"
    score = round(0.5 + 0.5 * (1 - (turn - 1) / max_turns), 6) if solved else 0.0
    if finished:
        step[0] += 1
        events.append({
            "type": "episode_end", "episode": current_ep, "target": target,
            "solved": solved, "turns": turn, "score": score, "best_rank": best,
            "best_rank_curve": curve, "lives_left": lives, "booms": booms,
            "warns": warns, "mines": mines, "max_plateau": 0, "fixation_sim": 0.0,
            "stop_reason": stop_reason, "ts": _ts(step[0]),
        })
    _write_lines(run_dir / "models" / slug / "events.jsonl", events)
    turn_evs = [e for e in events if e.get("type") == "turn"]
    le = turn_evs[-1] if turn_evs else {}
    live = {
        "model": model, "effort": effort, "episode": current_ep, "turn": turn,
        "max_turns": max_turns, "phase": phase,
        "last_guess": le.get("guess", ""), "last_similarity": le.get("similarity"),
        "last_rank": le.get("rank"), "best_rank": best, "lives": lives,
        "raw_snippet": f"GUESS {le.get('guess', '')}" if le.get("guess") else "",
        "updated_at": _ts(step[0]),
    }
    _write(run_dir / "models" / slug / "live.json", live)
    summary = {
        "model": model, "effort": effort, "episodes": [],
        "mean_score": score if finished else 0.0,
        "solve_rate": 1.0 if solved else 0.0, "median_turns": turn,
        "median_best_rank": best, "median_booms": booms,
        "mined_rate": 1.0 if mined else 0.0,
    }
    if usage_pt is not None:
        summary["usage"] = _usage(usage_pt[0] * turn, usage_pt[1] * turn)
    if finished:
        summary["episodes"] = [{
            "type": "episode_end", "episode": current_ep, "solved": solved,
            "turns": turn, "score": score, "best_rank": best,
            "best_rank_curve": curve, "lives_left": lives, "booms": booms,
            "warns": warns, "mines": mines, "max_plateau": 0, "fixation_sim": 0.0,
            "stop_reason": stop_reason, "target": target, "ts": _ts(step[0]),
        }]
    _write(run_dir / "models" / slug / "summary.json", summary)


def _build_rulelab(root) -> dict:
    """비밀 규칙 연구소: 규명 성공/부분 정답/실험 중/대기 참가자."""
    rid = "arena-rulelab"
    rule = "a×2+b"       # 진행 중엔 은닉, 완료 참가자 episode_end에만 공개
    rr = root / rid
    participants = [
        {"model": "claude-opus-4-8", "effort": "high", "slug": "claude-opus-4-8@high"},
        {"model": "claude-sonnet-5", "effort": "medium", "slug": "claude-sonnet-5@medium"},
        {"model": "codex-5.6-luna", "effort": "low", "slug": "codex-5.6-luna@low"},
        # 대기 중(디렉터리 없음) 참가자
        {"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high"},
    ]
    # 규명 성공: 실험 4회 후 5문항 전부 적중
    _rulelab_participant(rr, "claude-opus-4-8@high", "claude-opus-4-8", "high", [
        ("test", 3, 4, 10), ("test", 5, 5, 15), ("test", 10, 0, 20), ("test", 0, 7, 7),
        ("answer", [7, 13, 20, 31, 60], 5),
    ], finished=True, rule=rule, max_turns=15, phase="done", usage_pt=(7000, 160))
    # 부분 정답: 중복 실험 1 + 3/5 적중
    _rulelab_participant(rr, "claude-sonnet-5@medium", "claude-sonnet-5", "medium", [
        ("test", 1, 1, 3), ("test", 2, 2, 6), ("test", 1, 1, 3),
        ("answer", [7, 13, 20, 25, 55], 3),
    ], finished=True, rule=rule, max_turns=15, phase="done")
    # 실험 중(미답변): episode_end 없음 → 규칙 은닉
    _rulelab_participant(rr, "codex-5.6-luna@low", "codex-5.6-luna", "low", [
        ("test", 4, 4, 12), ("test", 8, 8, 24),
        ("fmt", "음... 곱셈이 섞인 것 같은데 확실치 않다"),
        ("test", 10, 10, 30), ("test", 0, 0, 0),
    ], finished=False, rule=rule, max_turns=15, phase="running", usage_pt=(9000, 120))
    _write_stream(rr, "codex-5.6-luna@low", {
        "model": "codex-5.6-luna", "effort": "low", "episode": 1, "turn": 5,
        "text": "0,0에서 0이 나왔으니 상수항은 없다. 이제 계수를 좁혀보자.\nTEST 1 0",
        "done": False, "updated_at": "2026-07-15T10:12:00",
    })
    _write(rr / "manifest.json", _manifest_game(
        rid, "ko-rulelab", participants, 1, 15, "running",
        "2026-07-15T10:05:00", None))
    return {"run_id": rid, "game": "ko-rulelab",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 15, "effort": None,
            "status": "running", "started_at": "2026-07-15T10:05:00",
            "finished_at": None}


def _build_maze(root) -> dict:
    """숨은 지도 탐험: 도착 성공(목표 공개) + 안개 탐사 중 2명(벽 충돌·형식 오류)."""
    rid = "arena-maze"
    target = "6,6"      # 진행 중엔 은닉, 도착(완료) 참가자만 공개
    mr = root / rid
    participants = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
        {"model": "codex-5.6-sol", "effort": "low", "slug": "codex-5.6-sol@low"},
    ]
    # 도착 성공: (0,0)→…→(6,6). dist 11→0.
    _maze_participant(mr, "claude-opus-4-8@low", "claude-opus-4-8", "low", [
        ("move", "동", [1, 0], "동·서", "남동", 11, True),
        ("move", "동", [2, 0], "남·동·서", "남동", 10, True),
        ("move", "남", [2, 1], "북·남", "남동", 9, True),
        ("move", "남", [2, 2], "북·동", "남동", 8, True),
        ("move", "동", [3, 2], "동·서", "남동", 7, True),
        ("move", "동", [4, 2], "남·서", "남동", 6, True),
        ("move", "남", [4, 3], "북·남", "남동", 5, True),
        ("move", "남", [4, 4], "북·동", "남동", 4, True),
        ("move", "동", [5, 4], "동·서", "남동", 3, True),
        ("move", "동", [6, 4], "남·서", "남", 2, True),
        ("move", "남", [6, 5], "북·남", "남", 1, True),
        ("move", "남", [6, 6], "북", "도착", 0, True),
    ], finished=True, target=target, max_turns=40, phase="done", shortest=12,
       usage_pt=(5000, 130))
    # 탐사 중: 벽 충돌 1회 포함, 미도착 → 목표 은닉
    _maze_participant(mr, "claude-sonnet-5@high", "claude-sonnet-5", "high", [
        ("move", "남", [0, 1], "북·남", "남동", 11, True),
        ("move", "남", [0, 2], "북·남", "남동", 10, True),
        ("move", "동", [0, 2], "북·남", "남동", 10, False),   # 벽에 막힘
        ("move", "남", [0, 3], "북·동", "남동", 9, True),
        ("move", "동", [1, 3], "남·서", "남동", 8, True),
    ], finished=False, target=target, max_turns=40, phase="running", usage_pt=(7000, 175))
    _write_stream(mr, "claude-sonnet-5@high", {
        "model": "claude-sonnet-5", "effort": "high", "episode": 1, "turn": 5,
        "text": "동쪽 벽에 한 번 막혔다. 남쪽·동쪽이 열려 있으니 목표 방위(남동)로 계속 간다.\nMOVE 동",
        "done": False, "updated_at": "2026-07-15T11:03:00",
    })
    # 탐사 중: 형식 오류 1회 포함
    _maze_participant(mr, "codex-5.6-sol@low", "codex-5.6-sol", "low", [
        ("move", "동", [1, 0], "동·서", "남동", 11, True),
        ("fmt", "북쪽으로 갈지 동쪽으로 갈지 고민된다"),
        ("move", "동", [2, 0], "남·동·서", "남동", 10, True),
    ], finished=False, target=target, max_turns=40, phase="running")
    _write(mr / "manifest.json", _manifest_game(
        rid, "ko-maze", participants, 1, 40, "running",
        "2026-07-15T11:00:00", None))
    return {"run_id": rid, "game": "ko-maze",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 40, "effort": None,
            "status": "running", "started_at": "2026-07-15T11:00:00",
            "finished_at": None}


def _build_minefield(root) -> dict:
    """의미 지뢰밭: 승리(목표+지뢰 공개) + 목숨 소진 패배 + 진행 중(은닉)."""
    rid = "arena-minefield"
    target = "사과"
    mines = ["전쟁", "질병"]     # 진행 중엔 은닉, 완료 참가자 episode_end에만 공개
    fr = root / rid
    participants = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
        {"model": "codex-5.6-luna", "effort": "low", "slug": "codex-5.6-luna@low"},
    ]
    # 승리: 정상 추측 + 경보 1회 후 목표 적중(목숨 3 유지)
    _mine_participant(fr, "claude-opus-4-8@low", "claude-opus-4-8", "low", [
        ("normal", "과일", 0.62, 12, None),
        ("normal", "바나나", 0.55, 25, 0.41),
        ("warn", "무기", 0.30, 60, 0.18),
        ("win", "사과"),
    ], finished=True, target=target, mines=mines, max_turns=40, phase="done",
       usage_pt=(6000, 145))
    # 패배: 경보 1 + 폭발 3 → 목숨 0(mined)
    _mine_participant(fr, "claude-sonnet-5@high", "claude-sonnet-5", "high", [
        ("warn", "감기", 0.40, 40, None),
        ("boom", "병", 2),
        ("boom", "바이러스", 1),
        ("boom", "전투", 0),
    ], finished=True, target=target, mines=mines, max_turns=40, phase="done")
    # 진행 중: 정상·중복·경보, 목숨 3 → episode_end 없음(목표·지뢰 은닉)
    _mine_participant(fr, "codex-5.6-luna@low", "codex-5.6-luna", "low", [
        ("normal", "딸기", 0.50, 30, None),
        ("dup", "딸기"),
        ("normal", "포도", 0.58, 18, 0.52),
        ("warn", "열매", 0.60, 14, 0.61),
    ], finished=False, target=target, mines=mines, max_turns=40, phase="running",
       usage_pt=(10000, 100))
    _write_stream(fr, "codex-5.6-luna@low", {
        "model": "codex-5.6-luna", "effort": "low", "episode": 1, "turn": 5,
        "text": "열매가 14위인데 지뢰 경보가 떴다. 방향은 맞지만 조심해서 우회하자.\nGUESS 사과나무",
        "done": False, "updated_at": "2026-07-15T12:04:00",
    })
    _write(fr / "manifest.json", _manifest_game(
        rid, "ko-minefield", participants, 1, 40, "running",
        "2026-07-15T12:00:00", None))
    return {"run_id": rid, "game": "ko-minefield",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 40, "effort": None,
            "status": "running", "started_at": "2026-07-15T12:00:00",
            "finished_at": None}


def _build_reuse(root) -> dict:
    """결과 재사용(측정 경제, 계약 §9): 새 모델만 실측(running) + 기존 참가자는 저장 결과 재사용.

    reused 참가자는 런 시작 시점부터 완료 상태(episode_end·summary·live phase="done")로
    존재한다 — manifest.participants[i].reused_from = 원본 run_id, manifest.measurement_key.
    fresh 참가자는 진행 중(episode_end 없음 → 정답 은닉). 배지 렌더 + 대기/중단 로직 무충돌 검증용.

    모델 주의: test_model_endpoint_participants의 정확-집합 단언(claude-haiku-4-5 → {LIVE,DONE},
    claude-opus-4-0 → 없음)을 피하려 그 두 모델은 쓰지 않는다 → opus-4-8@low, gemini-3-pro@high.
    """
    rid = "arena-fixture-reuse"
    src = "arena-fixture-done"          # 재사용 출처(자기 이전 런)
    target = "여름"                      # fresh는 진행 중 → 은닉, reused는 완료 → 자기 episode_end에서만 공개
    rr = root / rid
    participants = [
        # 신규 실측(fresh): 이번 런에서 새로 돌리는 모델 — 진행 중, 스레드 살아있음
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        # 재사용(reused): 저장 결과를 편입 — 런 시작부터 완료, reused_from 표식
        {"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high",
         "reused_from": src},
    ]
    # fresh: 진행 중(best 8), episode_end 없음 → 이 참가자 화면엔 정답 은닉
    _write_participant(rr, "claude-opus-4-8@low", "claude-opus-4-8", "low", {
        1: ([("v", "계절", 0.58, 30), ("v", "더위", 0.66, 12), ("v", "휴가", 0.70, 8)],
            False, target),
    }, current_ep=1, phase="running", max_turns=15)
    # reused: 런 시작부터 완료(episode_end/summary 존재, live phase=done), best 2(미해결)
    _write_participant(rr, "gemini-3-pro@high", "gemini-3-pro", "high", {
        1: ([("v", "봄", 0.50, 40), ("v", "장마", 0.68, 9), ("v", "무더위", 0.82, 2)],
            True, target),
    }, current_ep=1, phase="done", max_turns=15)
    man = _manifest_v2(rid, participants, 1, 15, "running",
                       "2026-07-15T14:00:00", None)
    # 측정 조건 키(엔진이 계산하는 sha256의 자리 표시자) — 웹은 이 필드를 표시하지 않고 통과만 한다.
    man["measurement_key"] = "reuse-fixture-key-0001"
    _write(rr / "manifest.json", man)
    _write_stream(rr, "claude-opus-4-8@low", {
        "model": "claude-opus-4-8", "effort": "low", "episode": 1, "turn": 3,
        "text": "휴가가 8위로 방향은 맞다. 계절·날씨 쪽 단어를 더 좁혀보자.\nGUESS 폭염",
        "done": False, "updated_at": "2026-07-15T14:03:00",
    })
    return {"run_id": rid, "game": "ko-semantle",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 15, "effort": None,
            "status": "running", "started_at": "2026-07-15T14:00:00",
            "finished_at": None}


def _build_tie(root) -> dict:
    """정렬 꼬리 검증 런(ko-semantle, 완료): 4명 전원 rank1·5턴으로 진행도(정답 근접)·턴 수가
    동률 → 오직 공통 꼬리(시간 ↑ → 출력 토큰 ↑ → slug)로만 순위가 갈린다.

    - sonnet@high: 시간 20s (최소) → 1등
    - codex@high : 시간 45s · 출력 700tok  (opus와 시간 동률, 토큰 적어 opus보다 앞)
    - opus@high  : 시간 45s · 출력 1200tok
    - gemini@low : usage 없음(구형) → 시간 단계에서 최하

    승자 tie-break(_run_summary)와 레인 자원 지표(시간·토큰) 둘 다 이 런에서 검증한다.
    """
    rid = "arena-fixture-tie"
    target = "여명"       # 완료 에피소드 → 정답 공개(은닉 대상 아님)
    tr = root / rid
    participants = [
        {"model": "claude-opus-4-8", "effort": "high", "slug": "claude-opus-4-8@high"},
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
        {"model": "codex-5.6-luna", "effort": "high", "slug": "codex-5.6-luna@high"},
        {"model": "gemini-3-pro", "effort": "low", "slug": "gemini-3-pro@low"},
    ]

    def solve5(w):    # 5턴 만에 정답(rank1) — 진행도·턴 수를 모두 동률로 고정
        return [("v", w[0], 0.55, 40), ("v", w[1], 0.62, 22), ("v", w[2], 0.70, 11),
                ("v", w[3], 0.78, 5), ("v", "여명", 1.0, 1)]

    _write_participant(tr, "claude-opus-4-8@high", "claude-opus-4-8", "high",
                       {1: (solve5(["빛", "새벽", "아침", "동틀녘"]), True, target)},
                       current_ep=1, phase="done", max_turns=15, usage_pt=(9000, 240))
    _write_participant(tr, "claude-sonnet-5@high", "claude-sonnet-5", "high",
                       {1: (solve5(["햇살", "여명기", "일출", "노을"]), True, target)},
                       current_ep=1, phase="done", max_turns=15, usage_pt=(4000, 180))
    _write_participant(tr, "codex-5.6-luna@high", "codex-5.6-luna", "high",
                       {1: (solve5(["여울", "여운", "여백", "여신"]), True, target)},
                       current_ep=1, phase="done", max_turns=15, usage_pt=(9000, 140))
    _write_participant(tr, "gemini-3-pro@low", "gemini-3-pro", "low",
                       {1: (solve5(["밝음", "여울목", "아침놀", "먼동"]), True, target)},
                       current_ep=1, phase="done", max_turns=15)   # usage 없음(구형)
    _write(tr / "manifest.json", _manifest_v2(
        rid, participants, 1, 15, "done",
        "2026-07-15T15:00:00", "2026-07-15T15:08:00"))
    return {"run_id": rid, "game": "ko-semantle",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 15, "effort": None,
            "status": "done", "started_at": "2026-07-15T15:00:00",
            "finished_at": "2026-07-15T15:08:00"}


def _cov_summary(run_dir, slug, model, effort, episodes, *, model_error=False,
                 mean_score=0.8, solve_rate=1.0, median_turns=5):
    """coverage 판정용 최소 summary.json(참가자 완주 여부만 결정).

    model_error=True면 첫 에피소드에 stop_reason='model_error'를 실어 '측정 완료'에서
    제외되게 한다(엔진과 동일 기준의 웹 판정 검증). mean_score/solve_rate/median_turns는
    모델별 탭의 플레이 가중 평균 검증용(런마다 다른 값을 주면 병합 평균이 각 런과 달라진다).
    """
    eps = []
    for i in range(1, episodes + 1):
        e = {"type": "episode_end", "episode": i, "solved": True,
             "turns": 5, "score": mean_score, "target": "x"}
        if model_error and i == 1:
            e["solved"] = False
            e["stop_reason"] = "model_error"
        eps.append(e)
    _write(run_dir / "models" / slug / "summary.json", {
        "model": model, "effort": effort, "episodes": eps,
        "mean_score": mean_score, "solve_rate": solve_rate, "median_turns": median_turns,
    })


def _cov_manifest(run_dir, rid, game, participants, episodes, max_turns,
                  status, started, finished, seeds):
    man = {"run_id": rid, "game": game,
           "models": [p["slug"] for p in participants],
           "participants": participants,
           "episodes": episodes, "max_turns": max_turns,
           "status": status, "started_at": started, "finished_at": finished}
    if seeds is not None:            # None이면 seeds 키 자체 없음('시드 기록 없음' 그룹)
        man["seeds"] = seeds
    _write(run_dir / "manifest.json", man)


def _seedcov_row(rid, game, participants, episodes, max_turns, status, started, finished):
    return {"run_id": rid, "game": game,
            "models": [p["slug"] for p in participants],
            "episodes": episodes, "max_turns": max_turns, "effort": None,
            "status": status, "started_at": started, "finished_at": finished}


def _build_seedcov(root) -> list:
    """시드별 기록 뷰(coverage) 검증 픽스처 — index에 추가할 런 row 리스트 반환.

    시드 777777·게임 ko-semantle 아래:
      - 1판×10턴 런 2개(A/B): 참가자 부분 겹침 → 측정 합집합 = {opus@low, sonnet@high, codex-luna@low}.
        run B의 codex-sol@low은 model_error 에피소드 → 완주에서 제외(coverage 미기여).
      - 1판×20턴 런 1개(C): 조건(턴수)이 달라 별도 행으로 분리.
    시드 없는 구형 런(L): 'seeds' 키 없음 → '시드 기록 없음' 그룹.
    모델 주의: test_model_endpoint_participants 정확 집합 단언을 피해 haiku-4-5·opus-4-0 미사용.
    """
    rows = []
    S = [777777]
    # A: opus@low, sonnet@high (둘 다 완주)
    ra, pa = "arena-seedcov-a", [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
    ]
    _cov_summary(root / ra, "claude-opus-4-8@low", "claude-opus-4-8", "low", 1)
    _cov_summary(root / ra, "claude-sonnet-5@high", "claude-sonnet-5", "high", 1)
    _cov_manifest(root / ra, ra, "ko-semantle", pa, 1, 10, "done",
                  "2026-07-15T18:00:00", "2026-07-15T18:05:00", S)
    rows.append(_seedcov_row(ra, "ko-semantle", pa, 1, 10, "done",
                             "2026-07-15T18:00:00", "2026-07-15T18:05:00"))
    # B: sonnet@high(겹침), codex-luna@low, codex-sol@low(model_error → 제외)
    rb, pb = "arena-seedcov-b", [
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
        {"model": "codex-5.6-luna", "effort": "low", "slug": "codex-5.6-luna@low"},
        {"model": "codex-5.6-sol", "effort": "low", "slug": "codex-5.6-sol@low"},
    ]
    _cov_summary(root / rb, "claude-sonnet-5@high", "claude-sonnet-5", "high", 1)
    _cov_summary(root / rb, "codex-5.6-luna@low", "codex-5.6-luna", "low", 1)
    _cov_summary(root / rb, "codex-5.6-sol@low", "codex-5.6-sol", "low", 1, model_error=True)
    _cov_manifest(root / rb, rb, "ko-semantle", pb, 1, 10, "done",
                  "2026-07-15T18:10:00", "2026-07-15T18:15:00", S)
    rows.append(_seedcov_row(rb, "ko-semantle", pb, 1, 10, "done",
                             "2026-07-15T18:10:00", "2026-07-15T18:15:00"))
    # C: 같은 시드·게임, 다른 조건(1판×20턴) → 행 분리
    rc, pc = "arena-seedcov-c", [
        {"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high"},
    ]
    _cov_summary(root / rc, "gemini-3-pro@high", "gemini-3-pro", "high", 1)
    _cov_manifest(root / rc, rc, "ko-semantle", pc, 1, 20, "done",
                  "2026-07-15T18:20:00", "2026-07-15T18:25:00", S)
    rows.append(_seedcov_row(rc, "ko-semantle", pc, 1, 20, "done",
                             "2026-07-15T18:20:00", "2026-07-15T18:25:00"))
    # R: 시드 314159·게임 ko-rulelab·1판×15턴 — [미측정 채우기] 프리필+조건잠금 검증용.
    #    두 참가자만 완주 → 카탈로그 대비 미측정 다수 → '미측정 채우기' 버튼 노출.
    #    (semantle 아닌 게임의 프리필: 게임 세그먼트·하단 문구·발사 body.game 동기화 검증.)
    RR = [314159]
    rr_id, prr = "arena-seedcov-rule", [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high"},
    ]
    _cov_summary(root / rr_id, "claude-opus-4-8@low", "claude-opus-4-8", "low", 1)
    _cov_summary(root / rr_id, "gemini-3-pro@high", "gemini-3-pro", "high", 1)
    _cov_manifest(root / rr_id, rr_id, "ko-rulelab", prr, 1, 15, "done",
                  "2026-07-15T19:00:00", "2026-07-15T19:06:00", RR)
    rows.append(_seedcov_row(rr_id, "ko-rulelab", prr, 1, 15, "done",
                             "2026-07-15T19:00:00", "2026-07-15T19:06:00"))
    # L: 시드 없는 구형 런 → '시드 기록 없음' 그룹
    rl, pl = "arena-seedcov-legacy", [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
    ]
    _cov_summary(root / rl, "claude-opus-4-8@low", "claude-opus-4-8", "low", 1)
    _cov_manifest(root / rl, rl, "ko-semantle", pl, 1, 8, "done",
                  "2026-07-14T10:00:00", "2026-07-14T10:05:00", None)   # 오래된 구형 런(우산 하단)
    rows.append(_seedcov_row(rl, "ko-semantle", pl, 1, 8, "done",
                             "2026-07-14T10:00:00", "2026-07-14T10:05:00"))
    # M: 반복 수만 다른 두 런 병합 시나리오 — 시드 909090·ko-semantle·50턴.
    #   MA: 1판(1에피소드), MB: 4회 반복(4에피소드). 뷰에서 한 행으로 병합돼야 한다.
    #   커버리지 합집합: MA{opus@low, sonnet@high} ∪ MB{opus@low, gemini@high}.
    #   opus@low는 두 런 모두 완주 → 모델별 탭에서 5플레이 평균(점수 (0.6·1+0.9·4)/5=0.84 → 84).
    MSEED = 909090
    ma, pma = "arena-seedcov-mrg-a", [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
    ]
    _cov_summary(root / ma, "claude-opus-4-8@low", "claude-opus-4-8", "low", 1, mean_score=0.6)
    _cov_summary(root / ma, "claude-sonnet-5@high", "claude-sonnet-5", "high", 1)
    _cov_manifest(root / ma, ma, "ko-semantle", pma, 1, 50, "done",
                  "2026-07-15T21:00:00", "2026-07-15T21:08:00", [MSEED])
    rows.append(_seedcov_row(ma, "ko-semantle", pma, 1, 50, "done",
                             "2026-07-15T21:00:00", "2026-07-15T21:08:00"))
    mb, pmb = "arena-seedcov-mrg-b", [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high"},
    ]
    _cov_summary(root / mb, "claude-opus-4-8@low", "claude-opus-4-8", "low", 4, mean_score=0.9)
    _cov_summary(root / mb, "gemini-3-pro@high", "gemini-3-pro", "high", 4)
    _cov_manifest(root / mb, mb, "ko-semantle", pmb, 4, 50, "done",
                  "2026-07-15T21:20:00", "2026-07-15T21:44:00", [MSEED, MSEED, MSEED, MSEED])
    rows.append(_seedcov_row(mb, "ko-semantle", pmb, 4, 50, "done",
                             "2026-07-15T21:20:00", "2026-07-15T21:44:00"))
    return rows


def _build_repeat(root) -> list:
    """반복 측정(같은 시드 N회) 검증 픽스처 — index row 리스트 반환.

    - 반복 런(seeds=[555555]*3): 같은 문제('노을')를 3회 풀어 안정성 측정 → 보드 '시도 1/2/3' 선택기.
    - 연속 시드 구형 런(seeds=[555555,555556,555557]): 같은 seed_base지만 서로 다른 문제 →
      반복 런과 '별도 그룹'이어야 한다(그룹 키가 seeds 리스트 전체 기준임을 검증).
    모델 주의: haiku-4-5·opus-4-0 미사용.
    """
    rows = []
    rid = "arena-fixture-repeat"
    rr = root / rid
    parts = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "high", "slug": "claude-sonnet-5@high"},
    ]
    tgt = "노을"

    def solve(w):   # 4턴 만에 정답(rank1)
        return [("v", w[0], 0.55, 30), ("v", w[1], 0.66, 12), ("v", w[2], 0.78, 4),
                ("v", "노을", 1.0, 1)]

    # opus: 세 시도 모두 정답(안정적)
    _write_participant(rr, "claude-opus-4-8@low", "claude-opus-4-8", "low", {
        1: (solve(["저녁", "황혼", "석양빛"]), True, tgt),
        2: (solve(["노랑", "하늘", "노을녘"]), True, tgt),
        3: (solve(["구름", "일몰", "땅거미"]), True, tgt),
    }, current_ep=3, phase="done", max_turns=15, usage_pt=(6000, 150))
    # sonnet: 시도 편차(1·3 정답, 2 미해결) — 안정성 낮음
    _write_participant(rr, "claude-sonnet-5@high", "claude-sonnet-5", "high", {
        1: (solve(["하늘", "구름", "석양"]), True, tgt),
        2: ([("v", "밤", 0.4, 90), ("v", "어둠", 0.45, 70), ("v", "빛", 0.6, 20)], True, tgt),
        3: (solve(["햇살", "노을빛", "저녁놀"]), True, tgt),
    }, current_ep=3, phase="done", max_turns=15, usage_pt=(9000, 240))
    _write(rr / "manifest.json", _manifest_v2(
        rid, parts, 3, 15, "done", "2026-07-15T19:00:00", "2026-07-15T19:20:00",
        seeds=[555555, 555555, 555555]))
    rows.append({"run_id": rid, "game": "ko-semantle",
                 "models": [p["slug"] for p in parts], "episodes": 3, "max_turns": 15,
                 "effort": None, "status": "done",
                 "started_at": "2026-07-15T19:00:00", "finished_at": "2026-07-15T19:20:00"})
    # 연속 시드 구형 런(같은 seed_base, 다른 문제) — 반복 런과 별도 그룹이어야 함
    rc = "arena-repeat-consec"
    pc = [{"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high"}]
    _cov_summary(root / rc, "gemini-3-pro@high", "gemini-3-pro", "high", 3)
    _cov_manifest(root / rc, rc, "ko-semantle", pc, 3, 15, "done",
                  "2026-07-15T19:30:00", "2026-07-15T19:40:00", [555555, 555556, 555557])
    rows.append(_seedcov_row(rc, "ko-semantle", pc, 3, 15, "done",
                             "2026-07-15T19:30:00", "2026-07-15T19:40:00"))
    return rows


def _build_exhaust(root) -> dict:
    """라이브 런에서 '턴 소진 실패' 강등 검증(ko-semantle, status=running).

    codex는 best_rank 2로 진행 중 두 참가자(opus best 3, sonnet best 8)보다 순위가
    좋지만, 스레드가 끝났고(live phase='done') 미해결이라 정렬 티어상 진행 중 아래로
    강등돼야 한다. 대기 참가자(gemini, 디렉터리 없음)는 맨 밑.

    누수 주의: codex는 episode_end가 없다(phase='done'만) → target 은닉 유지(진행 중
    참가자와 같은 에피소드의 정답이 새지 않는다). 이는 브리프의 두 번째 판정 조건
    ('live phase가 done/failed인데 미해결')을 그대로 재현한 것.
    모델 주의: test_model_endpoint_participants의 정확 집합 단언을 피해 claude-haiku-4-5·
    claude-opus-4-0을 쓰지 않는다.
    """
    rid = "arena-fixture-exhaust"
    target = "노을"     # ep1 — 진행 중/미완이라 어느 참가자도 공개하지 않음
    er = root / rid
    participants = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "medium", "slug": "claude-sonnet-5@medium"},
        {"model": "codex-5.6-luna", "effort": "low", "slug": "codex-5.6-luna@low"},
        # 대기 중(디렉터리 없음) — 티어 최하
        {"model": "gemini-3-pro", "effort": "high", "slug": "gemini-3-pro@high"},
    ]
    # 진행 중 A: best 3 (추측어에 target 부분문자열이 없도록 — 누수 검사 유효성 유지)
    _write_participant(er, "claude-opus-4-8@low", "claude-opus-4-8", "low", {
        1: ([("v", "저녁", 0.55, 20), ("v", "황혼", 0.66, 9), ("v", "석양빛", 0.78, 3)],
            False, target),
    }, current_ep=1, phase="running", max_turns=15, usage_pt=(9000, 200))
    # 진행 중 B: best 8
    _write_participant(er, "claude-sonnet-5@medium", "claude-sonnet-5", "medium", {
        1: ([("v", "하늘", 0.50, 30), ("v", "구름", 0.60, 15), ("v", "석양", 0.70, 8)],
            False, target),
    }, current_ep=1, phase="running", max_turns=15, usage_pt=(6000, 150))
    # 턴 소진 실패: best 2(진행 중보다 순위 좋음)지만 스레드 종료(phase done)·미해결 →
    #   episode_end 없음(target 은닉) → viewOf phase='done' → 티어 2로 강등.
    ranks = [40, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 2, 2]   # best 2, rank1 없음
    codex_specs = [("v", f"근접어{i}", round(0.40 + i * 0.03, 3), rk)
                   for i, rk in enumerate(ranks, 1)]
    _write_participant(er, "codex-5.6-luna@low", "codex-5.6-luna", "low", {
        1: (codex_specs, False, target),
    }, current_ep=1, phase="done", max_turns=15, usage_pt=(14000, 90))
    _write(er / "manifest.json", _manifest_v2(
        rid, participants, 1, 15, "running", "2026-07-15T17:00:00", None))
    # 진행 중 참가자엔 생성 텍스트도 흘려 라이브 느낌(선택)
    _write_stream(er, "claude-opus-4-8@low", {
        "model": "claude-opus-4-8", "effort": "low", "episode": 1, "turn": 3,
        "text": "석양빛이 3위로 꽤 가까워졌다. 하늘·저녁 계열을 더 좁혀보자.\nGUESS 저녁놀",
        "done": False, "updated_at": "2026-07-15T17:02:00",
    })
    return {"run_id": rid, "game": "ko-semantle",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 15, "effort": None,
            "status": "running", "started_at": "2026-07-15T17:00:00",
            "finished_at": None}


def _build_preparing(root) -> dict:
    """게임 준비 중(오라클 콜드 로딩) 런: 예비 manifest만 있고 models/ 디렉터리는 없다.

    사용자가 '새 플레이'를 누른 직후 상태 — 엔진이 build_game 전 예비 manifest를
    기록해 index에 즉시 노출되지만, 참가자 live.json은 아직 없다. index/상세 status는
    'preparing'. 이 상태가 '완료'로 오표기되지 않고 '준비 중'으로 정직히 렌더돼야 한다.
    모델 주의: test_model_endpoint_participants의 정확 집합 단언(claude-haiku-4-5 →
    {LIVE,DONE}, claude-opus-4-0 → 없음)을 피하려 그 두 모델은 쓰지 않는다.
    """
    rid = "arena-fixture-preparing"
    participants = [
        {"model": "claude-opus-4-8", "effort": "high", "slug": "claude-opus-4-8@high"},
        {"model": "claude-sonnet-5", "effort": "medium", "slug": "claude-sonnet-5@medium"},
        {"model": "codex-5.6-luna", "effort": "low", "slug": "codex-5.6-luna@low"},
    ]
    # manifest.json만 쓴다(models/ 디렉터리 없음). _write가 부모 디렉터리를 만든다.
    _write(root / rid / "manifest.json", _manifest_preparing(
        rid, "ko-semantle", participants, 2, 15, "혼합",
        "2026-07-15T16:00:00", [314159, 314160]))
    return {"run_id": rid, "game": "ko-semantle",
            "models": [p["slug"] for p in participants],
            "episodes": 2, "max_turns": 15, "effort": None,
            "status": "preparing", "started_at": "2026-07-15T16:00:00",
            "finished_at": None}


def _build_prep_failed(root) -> dict:
    """준비 단계에서 죽은 런(ollama 다운 등): 예비 manifest가 status='failed' + error로
    덮였다(build_game 도달 못 함 → models/ 없음, finished_at 존재).

    상태칩은 '중단', 보드는 정직하게 중단으로 렌더돼야 한다('완료' 오표기 금지).
    모델 주의: 위와 동일하게 claude-haiku-4-5·claude-opus-4-0을 쓰지 않는다.
    """
    rid = "arena-fixture-prep-failed"
    participants = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "gemini-3-pro", "effort": "low", "slug": "gemini-3-pro@low"},
    ]
    man = _manifest_preparing(rid, "ko-semantle", participants, 1, 15, "low",
                              "2026-07-15T15:40:00", [314159])
    man["status"] = "failed"
    man["finished_at"] = "2026-07-15T15:40:22"
    man["error"] = "ollama 임베딩 서버에 연결할 수 없습니다(오라클 로딩 실패)"
    _write(root / rid / "manifest.json", man)
    return {"run_id": rid, "game": "ko-semantle",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 15, "effort": None,
            "status": "failed", "started_at": "2026-07-15T15:40:00",
            "finished_at": "2026-07-15T15:40:22"}


def _build_dense20(root) -> dict:
    """20참가자 semantle 완료 런(dense 모드 검증). 다양한 best_rank 분포 + usage.

    타이포 확대·공간 재배분·레이아웃 시프트 불변을 dense(참가자 8+)에서 검증하는 대상.
    모델 주의: test_model_endpoint_participants 정확 집합 단언을 피해 haiku-4-5·opus-4-0 미사용.
    """
    rid = "arena-fixture-dense20"
    dr = root / rid
    combos = [
        ("claude-opus-4-8", "high"), ("claude-opus-4-8", "low"), ("claude-opus-4-6", "high"),
        ("claude-sonnet-5", "high"), ("claude-sonnet-5", "medium"), ("claude-sonnet-4-6", "high"),
        ("claude-fable-5", "high"), ("claude-fable-5", "low"),
        ("codex-5.6-luna", "high"), ("codex-5.6-luna", "low"), ("codex-5.6-sol", "high"),
        ("codex-5.6-sol", "medium"), ("codex-5.6-terra", "high"), ("codex-5.4-mini", "high"),
        ("codex-5.5", "high"), ("gemini-3-pro", "high"), ("gemini-3-pro", "low"),
        ("gemini-3.5-flash", "medium"), ("gpt-oss-120b", "medium"), ("gpt-oss-120b", "low"),
    ]  # 20개
    participants = [{"model": m, "effort": e, "slug": f"{m}@{e}"} for m, e in combos]
    target = "우주"
    ranks = [1, 1, 1, 2, 3, 5, 8, 12, 18, 25, 33, 44, 60, 77, 90, 110, 130, 150, 165, 178]
    for i, (m, e) in enumerate(combos):
        rk = ranks[i]
        solved = (rk == 1)
        specs = [("v", "별", 0.50, min(178, rk + 40)),
                 ("v", "하늘", 0.60, min(178, rk + 20)),
                 ("v", "공간", 0.70, min(178, rk + 5))]
        specs.append(("v", "우주", 1.0, 1) if solved else ("v", "은하", 0.75, rk))
        _write_participant(dr, f"{m}@{e}", m, e, {1: (specs, True, target)},
                           current_ep=1, phase="done", max_turns=12,
                           usage_pt=(4000 + i * 300, 100 + i * 20))
    _write(dr / "manifest.json", _manifest_v2(
        rid, participants, 1, 12, "done",
        "2026-07-15T20:00:00", "2026-07-15T20:12:00", seeds=[424242]))
    return {"run_id": rid, "game": "ko-semantle",
            "models": [p["slug"] for p in participants],
            "episodes": 1, "max_turns": 12, "effort": None, "status": "done",
            "started_at": "2026-07-15T20:00:00", "finished_at": "2026-07-15T20:12:00"}


def build(root) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ep1_target, ep2_target = "사진", "바다"  # ep2는 진행 중 → 어디에도 안 씀

    # ============ 런 L: v2 (참가자=모델×effort), 진행 중, 2 에피소드 ============
    live_id = "arena-fixture-live"
    participants = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-5", "effort": "medium", "slug": "claude-sonnet-5@medium"},
        {"model": "claude-haiku-4-5", "effort": "high", "slug": "claude-haiku-4-5@high"},
        {"model": "claude-haiku-4-5", "effort": "low", "slug": "claude-haiku-4-5@low"},
        {"model": "codex-5.6-luna", "effort": "low", "slug": "codex-5.6-luna@low"},
        # 대기 중(큐잉된) 참가자: manifest/index엔 있으나 아직 시작 안 해 디렉터리 없음.
        # 일부러 _write_participant를 호출하지 않는다 → _api_run이 플레이스홀더로 합집합.
        {"model": "claude-opus-4-6", "effort": "high", "slug": "claude-opus-4-6@high"},
    ]
    lr = root / live_id

    # opus@low: ep2 선두(best 3)
    _write_participant(lr, "claude-opus-4-8@low", "claude-opus-4-8", "low", {
        1: ([("v", "장면", 0.51, 40), ("v", "사진", 0.72, 6), ("v", "이미지", 0.68, 11),
             ("v", "그림", 0.63, 22), ("v", "화면", 0.60, 28)], True, ep1_target),
        2: ([("v", "강", 0.55, 33), ("v", "호수", 0.66, 12), ("v", "바닷물", 0.81, 3),
             ("v", "파도", 0.78, 5)], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15, usage_pt=(11000, 210))

    # sonnet@medium: ep1 해결(rank1), ep2 중위
    _write_participant(lr, "claude-sonnet-5@medium", "claude-sonnet-5", "medium", {
        1: ([("v", "그림", 0.58, 30), ("v", "장면", 0.62, 19), ("v", "이미지", 0.71, 8),
             ("v", "사진", 1.0, 1)], True, ep1_target),
        2: ([("v", "물", 0.60, 25), ("v", "강물", 0.64, 18), ("v", "바다", 0.70, 12)], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15, usage_pt=(6000, 150))

    # h4.5@high: 같은 모델 高 effort — 더 잘함(ep1 best 6, ep2 best 8)
    _write_participant(lr, "claude-haiku-4-5@high", "claude-haiku-4-5", "high", {
        1: ([("v", "사람", 0.64, 15), ("v", "장면", 0.66, 14), ("v", "사진기", 0.74, 6),
             ("v", "그림", 0.63, 22)], True, ep1_target),
        2: ([("v", "해변", 0.68, 16), ("v", "바닷가", 0.72, 8)], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15, usage_pt=(8000, 190))

    # h4.5@low: 같은 모델 低 effort — 덜 잘함(ep1 best 15 + 형식 무효, ep2 후위)
    #   usage_pt 없음 → usage 결측 구형 참가자(레인 자원 지표 생략 · 정렬 꼬리 최하)
    _write_participant(lr, "claude-haiku-4-5@low", "claude-haiku-4-5", "low", {
        1: ([("v", "사람", 0.64, 15), ("v", "남자", 0.55, 165), ("v", "인간", 0.59, 95),
             ("v", "개인", 0.62, 48), ("fmt", "GUESS 자\nGUESS 것")], True, ep1_target),
        2: ([("v", "육지", 0.42, 90), ("v", "해변", 0.63, 20), ("v", "모래", 0.5, 55),
             ("v", "흙", 0.45, 60), ("v", "땅", 0.48, 48), ("v", "지형", 0.5, 35)], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15)

    # codex-luna@low: ep1 best 40 + 중복/형식 무효, ep2 중위 + 방금 무효
    _write_participant(lr, "codex-5.6-luna@low", "codex-5.6-luna", "low", {
        1: ([("v", "풍경", 0.49, 70), ("v", "경치", 0.52, 55), ("dup", "풍경"),
             ("v", "이미지", 0.6, 40), ("fmt", "잘 모르겠지만 아마도 자연 풍경일 것 같습니다.")], True, ep1_target),
        2: ([("v", "바닷가", 0.67, 14), ("v", "해안", 0.69, 10), ("dup", "해안")], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15, usage_pt=(14000, 90))

    _write(lr / "manifest.json", _manifest_v2(
        live_id, participants, 2, 15, "running", "2026-07-14T18:20:00", None))

    # ---- 라이브 생성 텍스트(stream.json): 모든 상태를 노출 ----
    # opus@low: 토큰 스트리밍 중(부분, 단어 중간에서 잘림) — done:false
    _write_stream(lr, "claude-opus-4-8@low", {
        "model": "claude-opus-4-8", "effort": "low", "episode": 2, "turn": 5,
        "text": ("바닷물이 3위, 파도가 5위로 꽤 가까워졌다.\n"
                 "물 그 자체나 큰 물을 가리키는 말이 더 가까울 것 같다.\n"
                 "다음 후보로 '해"),
        "done": False, "updated_at": "2026-07-14T18:24:05",
    })
    # haiku@high: 토큰 스트리밍 중(부분) — done:false
    _write_stream(lr, "claude-haiku-4-5@high", {
        "model": "claude-haiku-4-5", "effort": "high", "episode": 2, "turn": 3,
        "text": ("해변보다 바닷가가 8위로 조금 더 가까웠다.\n"
                 "이번엔 물이나 바다에 더 직접적인 단어를 시도한다. '해"),
        "done": False, "updated_at": "2026-07-14T18:24:04",
    })
    # sonnet@medium: 턴 종료(전체 텍스트 확정) — done:true
    _write_stream(lr, "claude-sonnet-5@medium", {
        "model": "claude-sonnet-5", "effort": "medium", "episode": 2, "turn": 3,
        "text": "강물이 18위였으니 물 자체를 직접 노려보자.\nGUESS 바다",
        "done": True, "updated_at": "2026-07-14T18:23:40",
    })
    # codex-luna@low: 델타 없는 대기(codex/gemini는 토큰 델타가 없음) — text:"" done:false
    _write_stream(lr, "codex-5.6-luna@low", {
        "model": "codex-5.6-luna", "effort": "low", "episode": 2, "turn": 4,
        "text": "", "done": False, "updated_at": "2026-07-14T18:24:06",
    })
    # haiku@low: stream.json 아예 없음 → 서버가 {text:"",done:true} 반환(레인에 생성줄 없음)
    #   (일부러 아무것도 쓰지 않는다)

    # ============ 런 D: 레거시(participants 없음, dir=모델 id), 완료 ============
    done_id = "arena-fixture-done"
    done_models = ["claude-haiku-4-5", "codex-5.6-luna", "claude-opus-4-8", "gemini-3-pro"]
    dtar = "우주"
    dr = root / done_id
    _write_participant(dr, "claude-opus-4-8", "claude-opus-4-8", "low", {
        1: ([("v", "별", 0.6, 18), ("v", "하늘", 0.58, 25), ("v", "공간", 0.72, 4),
             ("v", "우주", 1.0, 1)], True, dtar),
    }, current_ep=1, phase="done", max_turns=12, legacy=True)
    _write_participant(dr, "claude-haiku-4-5", "claude-haiku-4-5", "low", {
        1: ([("v", "행성", 0.55, 30), ("v", "지구", 0.5, 44), ("dup", "행성"),
             ("v", "은하", 0.66, 9)], True, dtar),
    }, current_ep=1, phase="done", max_turns=12, legacy=True)
    _write_participant(dr, "codex-5.6-luna", "codex-5.6-luna", "low", {
        1: ([("v", "밤", 0.4, 88), ("v", "어둠", 0.38, 96),
             ("fmt", "정답을 특정하기 어렵습니다.")], True, dtar),
    }, current_ep=1, phase="done", max_turns=12, legacy=True)
    _write_participant(dr, "gemini-3-pro", "gemini-3-pro", "low", {
        1: ([("v", "과학", 0.35, 120), ("v", "미래", 0.3, 150)], True, dtar),
    }, current_ep=1, phase="done", max_turns=12, legacy=True)
    _write(dr / "manifest.json", _manifest_legacy(
        done_id, done_models, 1, 12, "low", "done",
        "2026-07-14T17:05:00", "2026-07-14T17:11:40"))

    # ============ 런 F: v2, 프로세스 전체 종료(embedding 소켓 타임아웃) — 고아 레인 ============
    # 실제 사고 재현: 런은 status="failed"로 끝났는데 참가자 live.json은 phase="running"
    # 상태로 남았다(고아). episode_end가 없어(finished=False) 정답 미공개 → 표시 계층이
    # '중단됨'으로 렌더해야 한다(순위/이력은 마지막 진실 그대로 유지).
    # 모델 주의: test_model_endpoint_participants의 정확 집합 단언을 피하려 claude-haiku-4-5·
    # claude-opus-4-0을 쓰지 않는다 → opus-4-8@low, sonnet-4-6@high.
    failed_id = "arena-fixture-failed"
    fparticipants = [
        {"model": "claude-opus-4-8", "effort": "low", "slug": "claude-opus-4-8@low"},
        {"model": "claude-sonnet-4-6", "effort": "high", "slug": "claude-sonnet-4-6@high"},
    ]
    ftar = "우물"   # ep1 진행 중(미완) → 어디에도 안 씀(target 은닉)
    fr = root / failed_id
    # opus@low: 유효 턴 몇 개(best 7) 후 고아 — live phase=running 남음(이력·최고순위 보존)
    _write_participant(fr, "claude-opus-4-8@low", "claude-opus-4-8", "low", {
        1: ([("v", "샘", 0.52, 40), ("v", "물", 0.66, 15), ("v", "지하수", 0.71, 7),
             ("v", "우물물", 0.69, 9)], False, ftar),
    }, current_ep=1, phase="running", max_turns=15)
    # sonnet@high: 유효 턴 몇 개(best 4) 후 고아
    _write_participant(fr, "claude-sonnet-4-6@high", "claude-sonnet-4-6", "high", {
        1: ([("v", "구멍", 0.55, 33), ("v", "물", 0.66, 15), ("v", "샘물", 0.73, 6),
             ("v", "지하수", 0.78, 4)], False, ftar),
    }, current_ep=1, phase="running", max_turns=15)
    _write(fr / "manifest.json", _manifest_v2(
        failed_id, fparticipants, 1, 15, "failed",
        "2026-07-15T09:02:00", "2026-07-15T09:14:30",
        failure="embedding socket timeout"))

    # ============ 신규 3게임 런(계약 §3~§5) ============
    rulelab_row = _build_rulelab(root)
    maze_row = _build_maze(root)
    minefield_row = _build_minefield(root)

    # ============ 결과 재사용 런(계약 §9 부록 A) ============
    reuse_row = _build_reuse(root)

    # ============ 정렬 꼬리(동률 tie-break) 검증 런 ============
    tie_row = _build_tie(root)

    # ============ 라이브 런 '턴 소진 실패' 강등 검증 런 ============
    exhaust_row = _build_exhaust(root)

    # ============ 시드별 기록 뷰(coverage) 검증 런들 ============
    seedcov_rows = _build_seedcov(root)

    # ============ 반복 측정(같은 시드 N회) + 연속 시드 구형 런 ============
    repeat_rows = _build_repeat(root)

    # ============ 20참가자 완료 런(dense 타이포/공간 검증) ============
    dense20_row = _build_dense20(root)

    # ============ 게임 준비 중(오라클 로딩) 런 + 준비 중 죽은 런 ============
    preparing_row = _build_preparing(root)
    prep_failed_row = _build_prep_failed(root)

    # ============ index.json (최신 앞) ============
    # 주의: test_index_schema가 runs[0]==LIVE를 단언한다. 실패 런·신규 게임 런은 연대순으론
    # 최신이지만 LIVE/DONE 뒤에 붙인다(runs[0]은 LIVE 유지). 신규 게임 런은 semantle 3런 뒤에.
    index = {"runs": [
        {"run_id": live_id, "game": "ko-semantle",
         "models": [p["slug"] for p in participants],
         "episodes": 2, "max_turns": 15, "effort": None,
         "status": "running", "started_at": "2026-07-14T18:20:00", "finished_at": None},
        {"run_id": done_id, "game": "ko-semantle", "models": done_models,
         "episodes": 1, "max_turns": 12, "effort": "low",
         "status": "done", "started_at": "2026-07-14T17:05:00",
         "finished_at": "2026-07-14T17:11:40"},
        {"run_id": failed_id, "game": "ko-semantle",
         "models": [p["slug"] for p in fparticipants],
         "episodes": 1, "max_turns": 15, "effort": None,
         "status": "failed", "started_at": "2026-07-15T09:02:00",
         "finished_at": "2026-07-15T09:14:30"},
        rulelab_row, maze_row, minefield_row, reuse_row, tie_row,
        exhaust_row, preparing_row, prep_failed_row, *seedcov_rows, *repeat_rows,
        dense20_row,
    ]}
    _write(root / "index.json", index)
    return root


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arena-fixture")
    build(out)
    print(f"fixture written to {out}")
