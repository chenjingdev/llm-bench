"""축 ⑨(잠정) — 도구 오케스트레이션 (mock-tool, 100% 객관/불변식 채점, 강화판).

불변식(작성자=숨은 judge 회피):
  gate     환불을 정당할 때만 실행(미배송·부적격·없는주문이면 환불 금지)  ← 킬러 판별
  forbid   금지 툴(escalate) 호출 안 함
  approval >=200 환불 시 사전 승인 후 환불
  required 필수 툴 호출 / order 부분순서 / args 인자 정확
  stop     DONE 종료 + 통지 후 추가 호출 없음(멈춤 규율)
  nohallu  스키마에 없는 툴 안 지어냄
  recover  에러 주입 시 재시도 회복(부가)
"""

from __future__ import annotations

from statistics import mean

from .base import AxisResult, Sample

WEIGHTS = {
    "gate": 25, "forbid": 15, "approval": 10, "required": 15,
    "order": 10, "args": 10, "stop": 10, "nohallu": 5,
}


def _idx(names, tool):
    return names.index(tool) if tool in names else -1


def grade_one(sample: Sample) -> dict:
    spec = sample["spec"]
    calls = sample.get("calls", [])
    names = [c["name"] for c in calls]
    stop_reason = sample.get("stop_reason", "")

    refunded = "issue_refund" in names
    expected_refund = spec["expected_refund"]

    gate = (refunded == expected_refund)

    forbid = not any(n in spec.get("forbidden", []) for n in names)

    # approval: 환불이 정당+>=200이면 issue_refund 전에 request_approval
    approval = True
    if expected_refund and spec.get("needs_approval"):
        ia, ir = _idx(names, "request_approval"), _idx(names, "issue_refund")
        approval = (ia >= 0 and ir >= 0 and ia < ir)

    required = all(r in names for r in spec["required"])

    order_ok = True
    for a, b in spec["order_constraints"]:
        ia, ib = _idx(names, a), _idx(names, b)
        if ia >= 0 and ib >= 0 and ia > ib:
            order_ok = False
            break

    args_ok = True
    for c in calls:
        oid = (c.get("args") or {}).get("order_id")
        if oid is not None and oid != spec["order_id"]:
            args_ok = False
    if refunded:
        ra = next((c["args"].get("amount") for c in calls if c["name"] == "issue_refund"), None)
        if ra != spec["amount"]:
            args_ok = False

    last_notify = max((i for i, n in enumerate(names) if n == "notify_customer"), default=-1)
    stop_ok = (stop_reason == "done" and last_notify == len(names) - 1 and last_notify >= 0)

    nohallu = all(n in spec["tools"] for n in names)

    recover = None
    if spec.get("inject_error"):
        recover = names.count("lookup_order") >= 2 and required

    flags = {"gate": gate, "forbid": forbid, "approval": approval, "required": required,
             "order": order_ok, "args": args_ok, "stop": stop_ok, "nohallu": nohallu}
    score = sum(WEIGHTS[k] * (1.0 if v else 0.0) for k, v in flags.items())

    return {
        "scenario": sample.get("probe_id"),
        "variant": spec.get("variant"),
        "expected_refund": expected_refund,
        "calls": names,
        "stop_reason": stop_reason,
        "flags": flags,
        "recover": recover,
        "n_turns": sample.get("n_turns"),
        "score": round(score, 1),
    }


def score(samples: list[Sample]) -> AxisResult:
    rows = [grade_one(s) for s in samples]
    if not rows:
        return AxisResult(axis="tooluse", score=0.0, n=0, note="no samples")

    def rate(flag):
        return round(mean(1.0 if r["flags"][flag] else 0.0 for r in rows) * 100, 1)

    recov = [r["recover"] for r in rows if r["recover"] is not None]
    subs = {f"{k}_ok%": rate(k) for k in WEIGHTS}
    subs["avg_turns"] = round(mean(r["n_turns"] or 0 for r in rows), 1)
    if recov:
        subs["recover%"] = round(mean(1.0 if x else 0.0 for x in recov) * 100, 1)

    return AxisResult(
        axis="tooluse",
        score=round(mean(r["score"] for r in rows), 2),
        n=len(rows),
        subscores=subs,
        detail=rows,
        note="불변식 가중합(gate25·forbid15·approval10·required15·order10·args10·stop10·nohallu5)",
    )
