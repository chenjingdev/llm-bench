"""ko-rulelab — 비밀 규칙 연구소.

숨은 결정론 규칙 f(a, b)->정수를 실험(TEST)으로 알아내고, 고정된 퀴즈 입력 5개의
출력을 한 번에 예측(ANSWER)하는 게임. 임베딩 오라클이 필요 없는 순수 결정론.

재생 검증 제약: 동일 seed로 reset한 게임에 저장된 raw를 다시 parse->step하면 저장
이벤트와 완전히 같아야 한다. 그래서 모든 난수는 reset()의 random.Random(seed)에서만
파생하며, render/step에는 시간·전역 난수 등 비결정론이 전혀 없다.
"""

from __future__ import annotations

import random
import re
import statistics

from .base import Action, GameState, instance_tag_line, new_nonce


# ----------------------------------------------------------------------
# 규칙 템플릿 풀 (도메인 전역, ≥12개, 파라미터화)
#
# 각 빌더는 random.Random을 받아 파라미터를 결정론적으로 뽑고 (fn, desc)를 돌려준다.
# fn: (a, b) -> int (a, b ∈ 0..20). desc: 사람이 읽는 한국어/수식 설명(GameState.secret).
# 모든 템플릿은 도메인 전역에서 |출력| ≤ 1,000,000 을 보장한다(최댓값은 수백 수준).
# 난이도 스펙트럼: 쉬움(선형 a+b+k) → 중간(곱/절댓값/최대·최소) → 어려움(조건부/조합).
# ----------------------------------------------------------------------

# --- 쉬움: 순수 선형. 실험 2~3번이면 계수·상수 식별 가능. -----------------
def _t_add_const(rng):
    k = rng.randint(1, 9)
    return (lambda a, b: a + b + k), f"a+b+{k}"


def _t_scale_a(rng):
    k = rng.randint(2, 5)
    return (lambda a, b: a * k + b), f"a×{k}+b"


def _t_scale_b(rng):
    k = rng.randint(2, 5)
    return (lambda a, b: a + b * k), f"a+b×{k}"


def _t_sub_const(rng):
    k = rng.randint(1, 9)
    return (lambda a, b: a - b + k), f"a-b+{k}"


def _t_two_coeffs(rng):
    k = rng.randint(2, 6)
    m = rng.randint(2, 6)
    return (lambda a, b: a * k + b * m), f"a×{k}+b×{m}"


# --- 중간: 비선형 단일 형태. 곱·절댓값·제곱·최대/최소. --------------------
def _t_mul_minus(rng):
    k = rng.randint(1, 20)
    return (lambda a, b: a * b - k), f"a×b-{k}"


def _t_absdiff(rng):
    k = rng.randint(2, 8)
    return (lambda a, b: abs(a - b) * k), f"|a-b|×{k}"


def _t_square_minus_b(rng):
    k = rng.randint(1, 5)
    return (lambda a, b: a * a - b * k), f"a²-b×{k}"


def _t_sum_scaled(rng):
    k = rng.randint(2, 7)
    return (lambda a, b: (a + b) * k), f"(a+b)×{k}"


def _t_max_scaled(rng):
    k = rng.randint(2, 9)
    return (lambda a, b: max(a, b) * k), f"max(a,b)×{k}"


def _t_min_plus_max(rng):
    k = rng.randint(1, 5)
    return (lambda a, b: min(a, b) * k + max(a, b)), f"min(a,b)×{k}+max(a,b)"


def _t_mul_plus_diff(rng):
    return (lambda a, b: a * b + a - b), "a×b+a-b"


# --- 어려움: 조건부 분기·제곱차. 양쪽 분기를 실험해야 식별 가능. -----------
def _t_square_diff(rng):
    return (lambda a, b: a * a - b * b), "a²-b²"


def _t_cond_greater(rng):
    return (lambda a, b: a * b if a > b else a + b), "a>b이면 a×b, 아니면 a+b"


def _t_cond_parity(rng):
    return (lambda a, b: a + b if a % 2 == 0 else a * 2 - b), \
        "a가 짝수면 a+b, 홀수면 a×2-b"


def _t_cond_threshold(rng):
    k = rng.randint(15, 30)
    return (lambda a, b: a + b if a + b <= k else (a + b) * 2), \
        f"a+b가 {k} 이하면 a+b, 초과면 (a+b)×2"


RULE_TEMPLATES = (
    _t_add_const, _t_scale_a, _t_scale_b, _t_sub_const, _t_two_coeffs,
    _t_mul_minus, _t_absdiff, _t_square_minus_b, _t_sum_scaled, _t_max_scaled,
    _t_min_plus_max, _t_mul_plus_diff,
    _t_square_diff, _t_cond_greater, _t_cond_parity, _t_cond_threshold,
)


# 'TEST'/'ANSWER'로 시작하는 줄과 그 뒤 인자를 분리 캡처한다(형식 오류 원인 구분용).
_ACTION_LINE = re.compile(r"^[ \t]*(TEST|ANSWER)\b[ \t]*(.*)$",
                          re.MULTILINE | re.IGNORECASE)
_INT = re.compile(r"^-?\d+$")

_DOMAIN_LO = 0
_DOMAIN_HI = 20
_QUIZ_N = 5


def _make_quiz(rng, n: int = _QUIZ_N) -> list[tuple[int, int]]:
    """서로 다른 입력쌍 n개를 rng에서 결정론적으로 뽑는다(도메인 0..20)."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < n:
        pair = (rng.randint(_DOMAIN_LO, _DOMAIN_HI),
                rng.randint(_DOMAIN_LO, _DOMAIN_HI))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


class RuleLab:
    id = "ko-rulelab"
    version = "1.1.0"   # 프롬프트에 에피소드 인스턴스 태그 추가(측정 조건 변경)
    DEFAULT_MAX_TURNS = 15

    # 프로토콜 확장 속성(계약 §1/§3).
    TURN_FIELDS = ("kind", "input", "output", "answer", "correct")
    INVALID_KEEP = ()
    LIVE_LAST_FIELDS = ("kind", "input", "output")
    RESULT_FIELDS = ("solved", "turns", "score", "experiments",
                     "duplicate_tests", "correct")
    needs_ollama = False

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self.max_turns = max_turns

    @property
    def metadata(self) -> dict:
        # 오라클 없는 순수 결정론 게임의 identity(재생 검증 대조 대상, 코드 고정 → 안정).
        return {
            "game": self.id,
            "version": self.version,
            "type": "deterministic-rule",
            "domain": f"{_DOMAIN_LO}..{_DOMAIN_HI}",
            "templates": len(RULE_TEMPLATES),
            "quiz_size": _QUIZ_N,
        }

    def reset(self, seed: int, nonce: str | None = None) -> GameState:
        rng = random.Random(seed)
        builder = rng.choice(RULE_TEMPLATES)
        fn, desc = builder(rng)
        quiz = _make_quiz(rng)
        # secret = 사람이 읽는 규칙 설명(episode_end에서만 공개). render엔 절대 노출 금지.
        state = GameState(self.id, self.version, seed, self.max_turns, desc)
        state.nonce = new_nonce() if nonce is None else nonce  # 에피소드 시작 시 1회 발급
        state.private["fn"] = fn
        state.private["quiz"] = quiz
        state.private["quiz_answers"] = [fn(a, b) for (a, b) in quiz]
        return state

    # --- render -------------------------------------------------------
    def render(self, state: GameState) -> str:
        rows = []
        for e in state.history:
            if not e.get("valid"):
                rows.append(f'{e["turn"]}. 형식 오류 — {e.get("error", "무효")}')
            elif e.get("kind") == "test":
                a, b = e["input"]
                rows.append(f'{e["turn"]}. 실험 {a} {b} → {e["output"]}')
            elif e.get("kind") == "answer":
                preds = " ".join(str(v) for v in e["answer"])
                rows.append(f'{e["turn"]}. 예측 {preds} → 정답 {e["correct"]}/5')
        history = "\n".join(rows) if rows else "아직 실험 없음"
        experiments = sum(1 for e in state.history
                          if e.get("valid") and e.get("kind") == "test")
        quiz_lines = "\n".join(
            f"{i + 1}) a={a}, b={b}"
            for i, (a, b) in enumerate(state.private["quiz"]))
        # 프롬프트 캐시 정렬(계약 §0): [고정 규칙 블록(퀴즈 포함, 에피소드 내 불변)]
        # → [이전 기록(append-only, 오래된 것부터)] → [변동부: 현재 턴/실험 수/행동 지시].
        # 규칙+기록은 턴이 지나도 바이트가 연장만 되므로 prefix 캐시가 히트한다.
        # 맨 앞 인스턴스 태그(에피소드 내 불변)는 레인별 프롬프트 비동일화용 고정부다.
        return (
            instance_tag_line(state) +
            "숨은 규칙 f(a, b)를 실험으로 알아내는 게임입니다.\n"
            "규칙은 두 정수 a, b(각각 0 이상 20 이하)를 받아 하나의 정수를 돌려줍니다.\n"
            "TEST로 원하는 입력을 넣어 출력을 관찰하고, 규칙의 정체를 추론하세요.\n"
            "규칙을 알아냈다고 판단하면 ANSWER로 아래 다섯 입력의 출력을 한꺼번에 예측합니다.\n\n"
            f"퀴즈 입력(이 다섯 쌍의 출력을 맞혀야 합니다):\n{quiz_lines}\n\n"
            "경고: ANSWER는 이 에피소드에서 단 한 번만 제출할 수 있고, "
            "제출하는 순간 에피소드가 끝납니다.\n"
            "매 응답에는 정확히 한 개의 행동만 담으세요.\n\n"
            "행동 형식(둘 중 하나를 정확히 한 줄로):\n"
            "TEST <a> <b>\n"
            "ANSWER <v1> <v2> <v3> <v4> <v5>\n\n"
            f"이전 기록:\n{history}\n\n"
            f"현재 턴: {state.turn + 1}/{state.max_turns}\n"
            f"지금까지 실험: {experiments}회\n\n"
            "다음 형식으로 한 개의 행동만 출력하세요.\n"
            "TEST <a> <b>  또는  ANSWER <v1> <v2> <v3> <v4> <v5>"
        )

    # --- parse --------------------------------------------------------
    def parse(self, text: str) -> Action:
        raw = text or ""
        matches = _ACTION_LINE.findall(raw)
        if len(matches) != 1:
            # 행동 줄이 0개 또는 2개 이상 — 실제 원인을 말한다.
            return Action("", "", raw, False,
                          "정확히 한 개의 행동(TEST 또는 ANSWER) 줄이 필요합니다")
        keyword, rest = matches[0]
        tokens = rest.split()
        if keyword.upper() == "TEST":
            if len(tokens) != 2 or not all(_INT.match(t) for t in tokens):
                return Action("test", "", raw, False,
                              "TEST 뒤에는 정수 두 개(a b)가 필요합니다")
            a, b = int(tokens[0]), int(tokens[1])
            if not (_DOMAIN_LO <= a <= _DOMAIN_HI and _DOMAIN_LO <= b <= _DOMAIN_HI):
                return Action("test", "", raw, False,
                              "입력 a, b는 0부터 20 사이의 정수여야 합니다")
            return Action("test", [a, b], raw)
        # ANSWER
        if len(tokens) != _QUIZ_N or not all(_INT.match(t) for t in tokens):
            return Action("answer", "", raw, False,
                          "ANSWER 뒤에는 정수 다섯 개(v1..v5)가 필요합니다")
        return Action("answer", [int(t) for t in tokens], raw)

    # --- step ---------------------------------------------------------
    def step(self, state: GameState, action: Action) -> dict:
        state.turn += 1
        if not action.valid:
            event = {"turn": state.turn, "valid": False, "raw": action.raw,
                     "error": action.error}
        elif action.kind == "test":
            a, b = action.value
            output = state.private["fn"](a, b)
            # 동일 입력 재실험도 유효 — 낭비(duplicate_tests)로만 측정한다.
            event = {"turn": state.turn, "valid": True, "raw": action.raw,
                     "kind": "test", "input": [a, b], "output": output}
        else:  # answer — 단 한 번의 기회, 제출 즉시 종료.
            preds = action.value
            answers = state.private["quiz_answers"]
            correct = sum(1 for p, ans in zip(preds, answers) if p == ans)
            event = {"turn": state.turn, "valid": True, "raw": action.raw,
                     "kind": "answer", "answer": list(preds), "correct": correct}
            state.done = True
            state.solved = (correct == _QUIZ_N)
            state.stop_reason = "answered"
        state.history.append(event)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return event

    # --- progress / result / summary ---------------------------------
    def progress(self, state: GameState) -> dict:
        experiments = sum(1 for e in state.history
                          if e.get("valid") and e.get("kind") == "test")
        answered = any(e.get("valid") and e.get("kind") == "answer"
                       for e in state.history)
        return {"experiments": experiments, "answered": answered}

    def result(self, state: GameState) -> dict:
        valid = [e for e in state.history if e.get("valid")]
        tests = [e for e in valid if e.get("kind") == "test"]
        answer_ev = next((e for e in valid if e.get("kind") == "answer"), None)

        # duplicate_tests: 동일 input 재실험 수(첫 등장은 제외, 재등장마다 +1).
        seen: set[tuple[int, int]] = set()
        duplicate = 0
        for e in tests:
            key = tuple(e["input"])
            if key in seen:
                duplicate += 1
            else:
                seen.add(key)

        # 미답변이면 correct=None(→ answer_rate 판별용), score=0.0.
        if answer_ev is not None:
            correct = answer_ev["correct"]
            score = correct / _QUIZ_N
        else:
            correct = None
            score = 0.0
        return {
            "solved": state.solved,
            "turns": state.turn,
            "score": round(score, 6),
            "experiments": len(tests),
            "duplicate_tests": duplicate,
            "correct": correct,
            "stop_reason": state.stop_reason,
        }

    def summary_stats(self, episode_ends: list[dict]) -> dict:
        if not episode_ends:
            return {"median_experiments": None, "answer_rate": None}
        exps = [e.get("experiments", 0) for e in episode_ends]
        answered = sum(1 for e in episode_ends if e.get("correct") is not None)
        return {
            "median_experiments": statistics.median(exps),
            "answer_rate": round(answered / len(episode_ends), 6),
        }
