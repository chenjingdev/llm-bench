"""러너 하네스 — probe × 모델 × N회를 같은 조건으로 실행하고 원시 응답을 JSONL로 저장.

원칙:
  · effort 고정 · 같은 호출 경로 · 설정 격리(client.py)
  · blind 라벨링: 모델에 A/B 라벨을 부여하고 매핑은 manifest에만 기록
    (객관 채점엔 영향 없지만, 향후 pairwise/judge 단계 재사용 대비)
  · 원시 응답은 results/raw/<run_id>/<axis>.jsonl 에 append
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import json as _json

from . import client, config, probes, scenarios


def _stamp() -> str:
    lt = time.localtime()
    return time.strftime("%Y%m%d-%H%M%S", lt)


def _singleturn_unit(model: str, probe: probes.Probe, repeat: int, effort: str) -> dict:
    """단일턴 축(density, instruction). meta를 레코드에 실어 채점기로 전달."""
    r = client.call(model, probe.prompt, effort=effort)
    return {
        "axis": probe.axis,
        "probe_id": probe.id,
        "model": model,
        "repeat": repeat,
        "prompt": probe.prompt,
        "text": r.text,
        "meta": probe.meta,
        "ok": r.ok,
        "error": r.error,
        "cost_usd": r.cost_usd,
        "output_tokens": r.output_tokens,
        "duration_ms": r.duration_ms,
    }


def _sycophancy_unit(model: str, probe: probes.Probe, repeat: int, effort: str) -> dict:
    """2라운드 에스컬레이션: 첫답 → 거짓압력 → (더 강한)거짓압력."""
    r1 = client.call(model, probe.prompt, effort=effort)
    r2 = client.call_followup(model, probe.prompt, r1.text, probe.challenge or "", effort=effort)
    ch2 = (probe.meta or {}).get("challenge2")
    if ch2:
        r3 = client.call_followup(model, probe.prompt, r2.text, ch2, effort=effort)
    else:
        r3 = None
    return {
        "axis": "sycophancy",
        "probe_id": probe.id,
        "model": model,
        "repeat": repeat,
        "prompt": probe.prompt,
        "answer": probe.answer,
        "challenge": probe.challenge,
        "challenge_valid": probe.challenge_valid,
        "meta": probe.meta,
        "text_first": r1.text,
        "text_followup": r2.text,
        "text_followup2": (r3.text if r3 else ""),
        "ok": r1.ok and r2.ok and (r3.ok if r3 else True),
        "error": (r1.error or r2.error or (r3.error if r3 else "")),
        "cost_usd": r1.cost_usd + r2.cost_usd + (r3.cost_usd if r3 else 0.0),
        "duration_ms": r1.duration_ms + r2.duration_ms + (r3.duration_ms if r3 else 0),
    }


def _render_chat(history: list[tuple[str, str]], latest: str) -> str:
    """누적 대화록을 단일 호출용 replay 트랜스크립트로 렌더(모델 자신의 직전 답이 닻)."""
    lines = ["[진행 중인 대화]"]
    for role, txt in history:
        who = "사용자" if role == "user" else "나"
        lines.append(f"{who}: {txt}")
    lines.append(f"사용자: {latest}")
    lines.append("\n위 대화의 맥락에서, 사용자의 마지막 메시지에 답하라.")
    return "\n".join(lines)


def _audience_unit(model: str, probe: probes.Probe, repeat: int, effort: str) -> dict:
    """청중적응(닻 감수성): cold / override / warm(잡담 닻+동일요청) 3조건 대조."""
    m = probe.meta
    cost, dur, ok, err = 0.0, 0, True, ""

    def _call(prompt):
        nonlocal cost, dur, ok, err
        r = client.call(model, prompt, effort=effort)
        cost += r.cost_usd; dur += r.duration_ms
        ok = ok and r.ok; err = err or r.error
        return r.text

    # A. cold — 맨정신 공식 문서
    cold_doc = _call(m["cold_req"])
    # C. override — 명시적 격식 명령(맨정신)
    override_doc = _call(m["override_req"])
    # B. warm — 반말 잡담으로 닻 내린 뒤 동일 요청. 모델 자신의 캐주얼 답이 history에 쌓임.
    history: list[tuple[str, str]] = []
    for anchor in m.get("warm_anchor", []):
        reply = _call(_render_chat(history, anchor))
        history.append(("user", anchor))
        history.append(("assistant", reply))
    warm_doc = _call(_render_chat(history, m["warm_doc"]))

    return {
        "axis": "audience",
        "probe_id": probe.id,
        "model": model,
        "repeat": repeat,
        "topic": m["topic"],
        "cold_req": m["cold_req"],
        "warm_doc_req": m["warm_doc"],
        "override_req": m["override_req"],
        "cold_doc": cold_doc,
        "warm_doc": warm_doc,
        "override_doc": override_doc,
        "warm_history": history,        # 닻 잡담의 모델 응답(디버그)
        "ok": ok,
        "error": err,
        "cost_usd": cost,
        "duration_ms": dur,
    }


def _tooluse_unit(model: str, sc: "scenarios.Scenario", repeat: int, effort: str,
                  max_turns: int = 12) -> dict:
    """mock-tool 다중턴 루프: CALL 파싱 → 각본 observation 되먹임 → DONE까지."""
    sysprompt = scenarios.system_prompt(sc)
    transcript = "[task]\n" + scenarios.initial_user(sc)
    calls: list[dict] = []
    trajectory: list[dict] = []
    cost = 0.0
    dur = 0
    stop_reason = "max_turns"
    ok = True

    for turn in range(max_turns):
        r = client.call(model, transcript, effort=effort, system=sysprompt)
        cost += r.cost_usd
        dur += r.duration_ms
        if not r.ok:
            stop_reason = f"error:{r.error[:40]}"
            ok = False
            break
        kind, a, b = scenarios.parse_action(r.text)
        step = {"turn": turn, "kind": kind, "text": r.text[:500]}
        if kind == "call":
            name, args = a, b
            obs = scenarios.simulate(sc, name, args, calls)
            calls.append({"name": name, "args": args})
            step["call"] = {"name": name, "args": args}
            step["obs"] = obs
            trajectory.append(step)
            transcript += (f"\n\n[you] CALL {name} {_json.dumps(args, ensure_ascii=False)}"
                           f"\n[observation] {_json.dumps(obs, ensure_ascii=False)}"
                           f"\n\n[continue: next CALL or DONE]")
        elif kind == "done":
            step["summary"] = a
            trajectory.append(step)
            stop_reason = "done"
            break
        else:
            trajectory.append(step)
            stop_reason = "no_action"  # 규약 미준수
            break

    return {
        "axis": "tooluse",
        "probe_id": sc.id,
        "model": model,
        "repeat": repeat,
        "spec": sc.spec,
        "calls": calls,
        "trajectory": trajectory,
        "stop_reason": stop_reason,
        "n_turns": len(trajectory),
        "ok": ok,
        "error": "" if ok else stop_reason,
        "cost_usd": cost,
        "duration_ms": dur,
    }


_UNIT = {"density": _singleturn_unit, "instruction": _singleturn_unit,
         "creativity": _singleturn_unit, "divergence": _singleturn_unit,
         "sycophancy": _sycophancy_unit,
         "tooluse": _tooluse_unit, "audience": _audience_unit}


def run(
    axes: list[str],
    models: list[str] | None = None,
    *,
    effort: str = config.DEFAULT_EFFORT,
    repeats: int = config.DEFAULT_REPEATS,
    workers: int = 3,
    limit: int | None = None,
    seed: int = 0,
) -> Path:
    """벤치 실행. run_id 디렉터리 경로를 반환."""
    config.ensure_dirs()
    config.env_guard()
    models = models or config.DEFAULT_MODELS

    run_id = _stamp()
    run_dir = config.RAW / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # blind 라벨: 모델 → A/B/... (실행마다 셔플)
    rng = random.Random(seed)
    labels = list("AB CDEFGH".replace(" ", ""))
    shuffled = models[:]
    rng.shuffle(shuffled)
    label_map = {m: labels[i] for i, m in enumerate(shuffled)}

    # 실행 단위(call unit) 평탄화
    jobs = []  # (axis, model, probe|scenario, repeat)
    for axis in axes:
        if axis == "tooluse":
            plist = scenarios.generate_set(seed=seed, n=6)
        else:
            plist = probes.build(axis, seed=seed)
        if limit:
            plist = plist[:limit]
        for model in models:
            for probe in plist:
                for rep in range(repeats):
                    jobs.append((axis, model, probe, rep))

    print(f"[run] id={run_id} models={[config.alias(m) for m in models]} "
          f"axes={axes} jobs={len(jobs)} effort={effort} workers={workers}")

    results: list[dict] = []
    t0 = time.time()

    def _do(job):
        axis, model, probe, rep = job
        rec = _UNIT[axis](model, probe, rep, effort)
        return axis, rec

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_do, j): j for j in jobs}
        for fut in as_completed(futs):
            axis, rec = fut.result()
            results.append(rec)
            # 축별 JSONL append
            with (run_dir / f"{axis}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            status = "ok" if rec["ok"] else f"ERR({rec['error'][:40]})"
            print(f"  [{done}/{len(jobs)}] {config.alias(rec['model'])} "
                  f"{rec['axis']}/{rec['probe_id']} {status} "
                  f"${rec['cost_usd']:.3f} {rec['duration_ms']}ms")

    total_cost = sum(r["cost_usd"] for r in results)
    n_err = sum(1 for r in results if not r["ok"])
    manifest = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "models": models,
        "label_map": label_map,
        "effort": effort,
        "repeats": repeats,
        "axes": axes,
        "seed": seed,
        "limit": limit,
        "n_jobs": len(jobs),
        "n_errors": n_err,
        "total_cost_usd": round(total_cost, 4),
        "wall_s": round(time.time() - t0, 1),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {run_id}  cost=${total_cost:.3f}  errors={n_err}  "
          f"wall={manifest['wall_s']}s  → {run_dir}")
    return run_dir
