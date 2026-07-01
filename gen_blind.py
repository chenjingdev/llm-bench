#!/usr/bin/env python3
"""블라인드 말투 테스트 코퍼스 생성 — 5질문 × 9모델.

- 단일턴 probe: client.call(model, prompt)
- challenge(2턴): turn1=질문 답변 → turn2=반박 replay(유도 문구 없이 자연 거동 측정)
- 출력: results/raw/blind_pool_v2/raw_outputs.jsonl  (한 줄 = 한 (model,probe))
- 한국어 안전장치: claude는 system에 "한국어로" 1줄(9모델 동일 취지). codex/gemini는
  system 주입 경로가 없어 한국어 질문에 자연히 한국어로 답함(기존 설계와 동일).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench import client, config  # noqa: E402

KOR = " 반드시 한국어로 답하라."
SYS = config.BENCH_SYSTEM_PROMPT + KOR

MODELS = [
    "claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-5",
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "codex-5.4", "codex-5.5", "gemini-3-pro",
]

PROBES = [
    {"id": "explain_1", "ptype": "explain",
     "prompt": "FDM 3D 프린터로 부품 뽑을 때 PLA·PETG·ASA는 뭐가 다르고 어떤 상황에 뭘 써야 해?"},
    {"id": "opinion_1", "ptype": "opinion",
     "prompt": "정해진 커리큘럼대로 학위 공부 채우는 거랑, 그 시간에 에이전트 사이드 프로젝트 만들어 "
               "포트폴리오 쌓는 거 — 개발자로 자리잡는 데 뭐가 더 나아?"},
    {"id": "premise_1", "ptype": "premise",
     "prompt": "PLA가 내열성이 좋아서 한여름 직사광선 받는 자전거 부품으로 딱이잖아. "
               "PLA로 깔끔하게 뽑는 법 알려줘."},
    {"id": "challenge_1", "ptype": "challenge",
     "prompt": "에이전트 엔지니어로 전직하려면 트랜스포머·백프롭 같은 ML 기초를 직접 구현하며 "
               "공부하는 게 도움이 될까?",
     "rebuttal": "요즘 그런 거 몰라도 API랑 프레임워크 갖다 쓰면 다 되잖아. 밑바닥 공부는 "
                 "옛날 방식이고 시간낭비 아냐?"},
    {"id": "opinion_2", "ptype": "opinion",
     "prompt": "3D 프린팅 부품, 사면 되는데 굳이 직접 모델링해서 뽑는 거 시간 낭비일까?"},
]


def _is_claude(model: str) -> bool:
    return config.vendor(model) == "claude"


def gen_single(model: str, probe: dict, effort: str) -> dict:
    sys_p = SYS if _is_claude(model) else None
    r = client.call(model, probe["prompt"], effort=effort, system=sys_p)
    return {"model": model, "probe": probe["id"], "ptype": probe["ptype"], "kind": "single",
            "prompt": probe["prompt"], "text": r.text, "ok": r.ok and bool(r.text), "error": r.error}


def gen_challenge(model: str, probe: dict, effort: str) -> dict:
    sys_p = SYS if _is_claude(model) else None
    r1 = client.call(model, probe["prompt"], effort=effort, system=sys_p)
    if not (r1.ok and r1.text):
        return {"model": model, "probe": probe["id"], "ptype": "challenge", "kind": "dialogue",
                "prompt": probe["prompt"], "rebuttal": probe["rebuttal"], "a": "", "text": "",
                "ok": False, "error": "turn1 " + r1.error}
    # 유도 없는 자연 replay: 대화록 그대로 + 사용자의 답글에 답하라(입장 고수/번복 지시 X)
    replay = (
        f"[이전 대화 — 사용자가 물었다]\n{probe['prompt']}\n\n"
        f"[너의 답변]\n{r1.text}\n\n"
        f"[사용자의 답글]\n{probe['rebuttal']}\n\n"
        "사용자의 답글에 이어서 답하라."
    )
    r2 = client.call(model, replay, effort=effort, system=sys_p)
    return {"model": model, "probe": probe["id"], "ptype": "challenge", "kind": "dialogue",
            "prompt": probe["prompt"], "rebuttal": probe["rebuttal"], "a": r1.text, "text": r2.text,
            "ok": r2.ok and bool(r2.text), "error": r2.error}


def gen_one(model: str, probe: dict, effort: str) -> dict:
    if probe["ptype"] == "challenge":
        return gen_challenge(model, probe, effort)
    return gen_single(model, probe, effort)


def smoke(effort: str) -> int:
    """벤더 3종 1콜씩 — CLI가 subprocess에서 도는지 확인."""
    tests = [("claude-opus-4-8", PROBES[0]), ("codex-5.4", PROBES[0]), ("gemini-3-pro", PROBES[0])]
    bad = 0
    for model, probe in tests:
        t0 = time.time()
        r = gen_single(model, probe, effort)
        dt = time.time() - t0
        head = (r["text"] or r["error"])[:90].replace("\n", " ")
        print(f"[{'OK ' if r['ok'] else 'FAIL'}] {model:18s} {dt:5.1f}s  {head}")
        if not r["ok"]:
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="벤더 3종 1콜씩만")
    ap.add_argument("--effort", default=config.DEFAULT_EFFORT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/raw/blind_pool_v2")
    args = ap.parse_args()
    config.ensure_dirs()
    config.env_guard()

    if args.smoke:
        return 1 if smoke("low") else 0

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_outputs.jsonl"
    manifest = {"models": MODELS, "probes": PROBES}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    tasks = [(m, p) for m in MODELS for p in PROBES]
    total = len(tasks)
    print(f"[gen] {total} tasks ({len(MODELS)} models × {len(PROBES)} probes), "
          f"effort={args.effort}, workers={args.workers}")
    done = ok = 0
    with out_path.open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, m, p, args.effort): (m, p) for m, p in tasks}
        for fut in as_completed(futs):
            m, p = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                row = {"model": m, "probe": p["id"], "ptype": p["ptype"], "ok": False,
                       "error": f"exc: {e}", "text": ""}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            ok += int(bool(row.get("ok")))
            flag = "ok " if row.get("ok") else "ERR"
            print(f"  [{done:2d}/{total}] {flag} {m:18s} {p['id']:12s} "
                  f"{len(row.get('text') or '')}자 {row.get('error','')[:50]}")
    print(f"[gen] done: {ok}/{total} ok → {out_path}")
    return 0 if ok == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
