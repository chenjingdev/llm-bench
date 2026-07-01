#!/usr/bin/env python3
"""blind_pool_v2 코퍼스 → blindrank.html DATA 블록 주입.

- 입력: results/raw/blind_pool_v2/{raw_outputs.jsonl, manifest.json}
- 발산점수/라벨: results/raw/blind_pool/live_result.json 재사용(모델 단위, probe 무관)
- 출력: blindrank.html 의 /* === DATA START === */ ~ /* === DATA END === */ 교체
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "results/raw/blind_pool_v2"
APP = ROOT / "blindrank.html"
LIVE = ROOT / "results/raw/blind_pool/live_result.json"

ALIAS = {
    "claude-opus-4-0": "Claude Opus 4.0", "claude-opus-4-1": "Claude Opus 4.1",
    "claude-opus-4-5": "Claude Opus 4.5", "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7", "claude-opus-4-8": "Claude Opus 4.8",
    "codex-5.4": "Codex GPT-5.4", "codex-5.5": "Codex GPT-5.5",
    "gemini-3-pro": "Gemini 3.1 Pro",
}


def main() -> int:
    manifest = json.loads((V2 / "manifest.json").read_text())
    models = manifest["models"]
    probes_raw = manifest["probes"]

    items = []
    seen = set()
    for ln in (V2 / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if not o.get("ok"):
            print(f"[skip] {o.get('model')} {o.get('probe')} ok=False ({o.get('error','')[:60]})")
            continue
        it = {"model": o["model"], "probe": o["probe"], "ptype": o["ptype"],
              "kind": o.get("kind", "single"), "text": o.get("text", ""), "ok": True}
        if o.get("kind") == "dialogue":
            it["a"] = o.get("a", "")
        items.append(it)
        seen.add((o["model"], o["probe"]))

    # 커버리지 점검
    expect = len(models) * len(probes_raw)
    print(f"[cov] items={len(items)}/{expect}  models={len(models)} probes={len(probes_raw)}")
    for p in probes_raw:
        miss = [m for m in models if (m, p["id"]) not in seen]
        if miss:
            print(f"  [warn] {p['id']}: 누락 {miss}")

    probes = [{"id": p["id"], "ptype": p["ptype"], "prompt": p["prompt"],
               **({"rebuttal": p["rebuttal"]} if p.get("rebuttal") else {})}
              for p in probes_raw]

    # 발산점수 + 라벨
    divergence, labels = {}, {}
    if LIVE.exists():
        live = json.loads(LIVE.read_text())
        divergence = live.get("divergence", {}) or {}
        labels = live.get("labels", {}) or {}
    for m in models:
        labels.setdefault(m, ALIAS.get(m, m))
        divergence.setdefault(m, None)

    data = {"models": models, "labels": labels, "divergence": divergence,
            "probes": probes, "items": items}
    block = "const BLIND_DATA = " + json.dumps(data, ensure_ascii=False) + ";"

    html = APP.read_text(encoding="utf-8")
    repl = "/* === DATA START === */\n" + block + "\n/* === DATA END === */"
    # 함수형 치환: block 내부의 \n(JSON 이스케이프)을 re.sub가 다시 해석하지 않게 한다
    new = re.sub(r"/\* === DATA START === \*/.*?/\* === DATA END === \*/",
                 lambda _m: repl, html, flags=re.S)
    if new == html:
        print("[err] DATA 마커를 못 찾음")
        return 1
    APP.write_text(new, encoding="utf-8")
    kb = len(new) // 1024
    print(f"[ok] 주입 완료 → blindrank.html ({kb}KB, items {len(items)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
