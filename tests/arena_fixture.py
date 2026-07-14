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


# --- 턴 스펙 DSL --------------------------------------------------------
# ("v", 단어, 유사도, 순위)       유효 추측
# ("dup", 단어)                    무효: 중복 추측(guess 있음)
# ("fmt", 원문)                    무효: 형식 오류(guess 없음)
def _build_episode(events, episode, specs, *, finished, target, step_ref):
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
                       current_ep, phase, max_turns, legacy=False):
    """참가자 하나(디렉토리=slug)를 쓴다.

    legacy=True면 live/summary에 effort를 넣지 않는다(레거시 파생 경로 테스트).
    """
    events = []
    step = [hash(slug) % 1000]
    ep_results = {}
    finished_results = []
    for ep in sorted(episodes):
        specs, finished, target = episodes[ep]
        r = _build_episode(events, ep, specs, finished=finished, target=target, step_ref=step)
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
    _write(run_dir / "models" / slug / "summary.json", summary)


def _oracle():
    return {"game": "ko-semantle", "version": "1.0.0",
            "type": "exact-cosine-reference-rank",
            "embedding_model": "qwen3-embedding:4b",
            "reference_words": REF_WORDS,
            "rank_scope": "pinned-reference-vocabulary"}


def _manifest_v2(run_id, participants, episodes, max_turns, status, started, finished,
                 failure=None):
    slugs = [p["slug"] for p in participants]
    man = {
        "run_id": run_id, "game": "ko-semantle", "models": slugs,
        "participants": participants,
        "episodes": episodes, "max_turns": max_turns,
        "status": status, "started_at": started, "finished_at": finished,
        "game_version": "1.0.0", "oracle": _oracle(), "seeds": [1784015657, 42],
        "verify": {"ok": True, "models": {s: {"ok": True, "errors": []} for s in slugs}},
    }
    if failure is not None:          # 중단 런에만 사유를 싣는다(정상 런엔 키 없음).
        man["failure"] = failure
    return man


def _manifest_legacy(run_id, models, episodes, max_turns, effort, status, started, finished):
    return {
        "run_id": run_id, "game": "ko-semantle", "models": models,
        "episodes": episodes, "max_turns": max_turns, "effort": effort,
        "status": status, "started_at": started, "finished_at": finished,
        "game_version": "1.0.0", "oracle": _oracle(), "seeds": [1784015657],
        "verify": {"ok": True, "models": {m: {"ok": True, "errors": []} for m in models}},
    }


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
    }, current_ep=2, phase="running", max_turns=15)

    # sonnet@medium: ep1 해결(rank1), ep2 중위
    _write_participant(lr, "claude-sonnet-5@medium", "claude-sonnet-5", "medium", {
        1: ([("v", "그림", 0.58, 30), ("v", "장면", 0.62, 19), ("v", "이미지", 0.71, 8),
             ("v", "사진", 1.0, 1)], True, ep1_target),
        2: ([("v", "물", 0.60, 25), ("v", "강물", 0.64, 18), ("v", "바다", 0.70, 12)], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15)

    # h4.5@high: 같은 모델 高 effort — 더 잘함(ep1 best 6, ep2 best 8)
    _write_participant(lr, "claude-haiku-4-5@high", "claude-haiku-4-5", "high", {
        1: ([("v", "사람", 0.64, 15), ("v", "장면", 0.66, 14), ("v", "사진기", 0.74, 6),
             ("v", "그림", 0.63, 22)], True, ep1_target),
        2: ([("v", "해변", 0.68, 16), ("v", "바닷가", 0.72, 8)], False, ep2_target),
    }, current_ep=2, phase="running", max_turns=15)

    # h4.5@low: 같은 모델 低 effort — 덜 잘함(ep1 best 15 + 형식 무효, ep2 후위)
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
    }, current_ep=2, phase="running", max_turns=15)

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

    # ============ index.json (최신 앞) ============
    # 주의: test_index_schema가 runs[0]==LIVE를 단언한다. 실패 런은 연대순으론 최신이지만
    # LIVE/DONE 뒤에 붙인다(runs[0]은 LIVE 유지).
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
    ]}
    _write(root / "index.json", index)
    return root


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arena-fixture")
    build(out)
    print(f"fixture written to {out}")
