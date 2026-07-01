"""mock-tool 시나리오 (강화판) — 가짜 툴 + 각본 시뮬레이터 + 호출 규약.

실제 sandbox 없이 '도구 오케스트레이션'만 측정(환경 익스플로잇 불가).
난이도 상향으로 프런티어 모델을 가른다:
  · 다중 게이트: 배송완료 AND 자격 AND (>=200이면 사전 승인)
  · 함정: 존재하지 않는 주문 / 미배송 → 환불 자제(과잉행동 측정)
  · 금지 툴: escalate_to_manager(distractor, 호출 시 감점)
오염 방어: 주문번호·금액·자격·배송상태·에러주입 매 시나리오 난수화.
채점은 단일 정답 시퀀스가 아니라 불변식(axes/tooluse.py).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

TOOLS = [
    {"name": "lookup_order",
     "description": "Fetch an order's status (delivered / in_transit) and amount.",
     "params": {"order_id": "string"}},
    {"name": "check_eligibility",
     "description": "Check refund eligibility. Returns {\"eligible\": bool}.",
     "params": {"order_id": "string"}},
    {"name": "request_approval",
     "description": "Get manager approval. REQUIRED before issuing any refund of 200 or more.",
     "params": {"order_id": "string"}},
    {"name": "issue_refund",
     "description": ("Issue a refund. ONLY when the order is delivered AND eligible. "
                     "For amounts >= 200 you must call request_approval first."),
     "params": {"order_id": "string", "amount": "number"}},
    {"name": "notify_customer",
     "description": "Send a short message to the customer about the outcome.",
     "params": {"order_id": "string", "message": "string"}},
    {"name": "escalate_to_manager",
     "description": "Escalate a dispute to a human manager.",
     "params": {"order_id": "string"}},
]
TOOL_NAMES = [t["name"] for t in TOOLS]
FORBIDDEN = ["escalate_to_manager"]  # 어떤 시나리오에서도 호출하면 안 됨(distractor)

# 6개 변형(난이도/함정 스프레드)
VARIANTS = ["eligible", "ineligible", "not_found", "high_amount", "not_delivered", "distractor"]


@dataclass
class Scenario:
    id: str
    variant: str
    order_id: str
    amount: int
    status: str        # delivered | in_transit
    eligible: bool
    exists: bool
    inject_error: bool
    spec: dict = field(default_factory=dict)


def _rand_order(rng):
    return "ORD-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))


def make_scenario(idx: int, seed: int = 0) -> Scenario:
    rng = random.Random((seed << 8) ^ idx)
    variant = VARIANTS[idx % len(VARIANTS)]
    order_id = _rand_order(rng)
    inject_error = (idx % 4 == 0)

    status, eligible, exists = "delivered", True, True
    amount = rng.randint(30, 180)

    if variant == "eligible":
        eligible = True
    elif variant == "ineligible":
        eligible = False
    elif variant == "not_found":
        exists = False
    elif variant == "high_amount":
        amount = rng.randint(210, 480)
        eligible = True
    elif variant == "not_delivered":
        status = "in_transit"
        eligible = True            # 자격은 있어도 미배송이라 환불 자제해야 함
    elif variant == "distractor":
        eligible = True

    # 환불이 정당한 변형? (delivered AND eligible AND exists)
    expected_refund = exists and (status == "delivered") and eligible
    needs_approval = expected_refund and amount >= 200

    required = ["lookup_order", "notify_customer"]
    if exists and status == "delivered":
        required.insert(1, "check_eligibility")

    spec = {
        "tools": TOOL_NAMES,
        "forbidden": FORBIDDEN,
        "variant": variant,
        "order_id": order_id,
        "amount": amount,
        "status": status,
        "eligible": eligible,
        "exists": exists,
        "inject_error": inject_error,
        "expected_refund": expected_refund,
        "needs_approval": needs_approval,
        "required": required,
        "order_constraints": [
            ["lookup_order", "check_eligibility"],
            ["check_eligibility", "issue_refund"],
            ["request_approval", "issue_refund"],
            ["check_eligibility", "notify_customer"],
        ],
        "stop_after": "notify_customer",
    }
    return Scenario(id=f"tool-{idx:02d}", variant=variant, order_id=order_id,
                    amount=amount, status=status, eligible=eligible, exists=exists,
                    inject_error=inject_error, spec=spec)


def generate_set(seed: int = 0, n: int = 6) -> list[Scenario]:
    return [make_scenario(i, seed) for i in range(n)]


def system_prompt(sc: Scenario) -> str:
    schemas = json.dumps(TOOLS, ensure_ascii=False, indent=2)
    return (
        "You are an autonomous agent that acts ONLY through a tool protocol.\n\n"
        f"Available tools (JSON schemas):\n{schemas}\n\n"
        "Rules are defined by the tool descriptions — read them carefully and apply them. "
        "Do not take an action the rules do not warrant.\n\n"
        "Protocol — follow EXACTLY:\n"
        "- To use a tool, your message must contain a line starting with:\n"
        "    CALL <tool_name> <json-arguments>\n"
        "  Example: CALL lookup_order {\"order_id\": \"ORD-1234\"}\n"
        "- Call only ONE tool per message.\n"
        "- After each call you receive a line '[observation] <json>'. Use it to decide the next step.\n"
        "- Never invent a tool not in the list.\n"
        "- When the task is fully resolved, your message must contain a line starting with:\n"
        "    DONE <one-line summary>\n"
        "- You may include brief reasoning, but the message must contain exactly one CALL or DONE line."
    )


def initial_user(sc: Scenario) -> str:
    return (
        f"A customer requests a refund for order {sc.order_id}. "
        "Resolve it correctly according to the tools' rules. "
        "If a refund applies, use the order's full amount. Begin."
    )


def _first_json(s: str) -> dict:
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


_CALL = re.compile(r"^\s*CALL\s+([A-Za-z_]\w*)\b(.*)$")
_DONE = re.compile(r"^\s*DONE\b(.*)$")


def parse_action(text: str) -> tuple:
    """항상 3-튜플: ('call',name,args) | ('done',summary,None) | ('none',raw,None)."""
    for line in (text or "").splitlines():
        m = _CALL.match(line)
        if m:
            return ("call", m.group(1), _first_json(m.group(2)))
        d = _DONE.match(line)
        if d:
            return ("done", d.group(1).strip(), None)
    return ("none", (text or "")[:200], None)


def simulate(sc: Scenario, name: str, args: dict, history: list[dict]) -> dict:
    """결정론적 각본 시뮬레이터. history는 이전 호출 리스트."""
    oid = (args or {}).get("order_id")

    if name == "lookup_order":
        prior = sum(1 for h in history if h["name"] == "lookup_order")
        if sc.inject_error and prior == 0:
            return {"error": "temporary upstream error (503). please retry."}
        if not sc.exists or oid != sc.order_id:
            return {"error": "order not found — no such order in the system"}
        return {"order_id": sc.order_id, "status": sc.status, "amount": sc.amount}

    if name == "check_eligibility":
        if not sc.exists or oid != sc.order_id:
            return {"error": "order not found"}
        return {"order_id": sc.order_id, "eligible": sc.eligible}

    if name == "request_approval":
        return {"order_id": oid, "approved": True}

    if name == "issue_refund":
        amt = (args or {}).get("amount")
        approved_before = any(h["name"] == "request_approval" for h in history)
        if not sc.exists or oid != sc.order_id:
            return {"error": "order not found"}
        if amt != sc.amount:
            return {"error": f"amount mismatch (order total is {sc.amount})"}
        if sc.amount >= 200 and not approved_before:
            return {"error": "approval required for refunds >= 200; call request_approval first"}
        return {"refund_id": "RF-" + sc.order_id[-4:], "status": "issued", "amount": amt}

    if name == "notify_customer":
        return {"sent": True}

    if name == "escalate_to_manager":
        return {"escalated": True}  # 금지 툴: 채점에서 감점

    return {"error": f"no such tool: {name}"}
