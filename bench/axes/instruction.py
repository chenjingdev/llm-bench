"""축 ② 지시 장악 — 쌓인 제약 충족률(judge-free, 100% 객관)."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .. import constraints
from .base import AxisResult, Sample


def grade_one(sample: Sample) -> dict:
    text = sample.get("text", "") or ""
    specs = sample.get("meta", {}).get("constraints", [])
    results = []
    for s in specs:
        ok = constraints.check(s["type"], text, s["params"])
        results.append({"type": s["type"], "ok": bool(ok)})
    passed = sum(1 for r in results if r["ok"])
    return {
        "probe_id": sample.get("probe_id"),
        "n_constraints": len(specs),
        "passed": passed,
        "frac": (passed / len(specs)) if specs else 0.0,
        "results": results,
        "all_ok": passed == len(specs) and bool(specs),
    }


def score(samples: list[Sample]) -> AxisResult:
    rows = [grade_one(s) for s in samples if s.get("meta", {}).get("constraints")]
    if not rows:
        return AxisResult(axis="instruction", score=0.0, n=0, note="no samples")

    # 제약 타입별 통과율(어디서 무너지나)
    by_type = defaultdict(lambda: [0, 0])  # type -> [pass, total]
    for r in rows:
        for c in r["results"]:
            by_type[c["type"]][1] += 1
            by_type[c["type"]][0] += 1 if c["ok"] else 0
    type_rates = {t: round(p / n * 100, 1) for t, (p, n) in by_type.items()}

    return AxisResult(
        axis="instruction",
        score=round(mean(r["frac"] for r in rows) * 100, 2),
        n=len(rows),
        subscores={
            "all_ok%": round(mean(1.0 if r["all_ok"] else 0.0 for r in rows) * 100, 1),
            "avg_constraints": round(mean(r["n_constraints"] for r in rows), 1),
            "by_type%": type_rates,
        },
        detail=rows,
        note="score = 제약 충족률(쌓인 제약 중 지킨 비율 평균)",
    )
