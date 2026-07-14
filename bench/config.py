"""전역 설정 — 비교 대상 모델, effort 고정, 경로, 격리 정책."""

from __future__ import annotations

import os
from pathlib import Path

# --- 경로 ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"          # 모델 원시 응답 JSONL
REPORTS = RESULTS / "reports"  # 집계/리포트 산출물

# --- 비교 대상 ----------------------------------------------------------
# 1차 가설: 4.8 = 실행/추론, 4.6 = 대화/발상.
# 호출 가능한 Opus 사다리: 4.0 · 4.1 · 4.5 · 4.6 · 4.7 · 4.8
DEFAULT_MODELS = ["claude-opus-4-8", "claude-opus-4-6"]

# --- 교차 vendor 모델(코덱스·agy 게이트웨이) ---------------------------
# 내부 id → 실제 CLI 모델명. 호출 경로는 client.py가 vendor로 라우팅.
#   codex-* → `codex exec -m <gpt-x>` (ChatGPT 구독). effort는 CLI 인자.
#   agy-*   → `agy -p --model "<표시명> (Effort)"` (effort가 표시명 접미사).
# 실호출 검증 완료(사용자 지시). 등록 금지:
#   ChatGPT 계정 미지원 — gpt-5.6 / gpt-5.6-mini / gpt-5.5-mini
#     ("not supported ... with a ChatGPT account").
#   존재하지 않는 이름(오타/추측) — gpt-5.6-tera(r 하나) / gpt-tera / gpt-5.7-tera /
#     gpt-terra / gpt-5.5-terra. 정식은 gpt-5.6-terra(r 두 개, 천체 luna/sol/terra).
CODEX_MODELS = {
    "codex-5.5": "gpt-5.5",
    "codex-5.4": "gpt-5.4",
    "codex-5.6-luna": "gpt-5.6-luna",
    "codex-5.6-sol": "gpt-5.6-sol",       # 사용자 주력
    "codex-5.6-terra": "gpt-5.6-terra",   # 천체 명명(r 두 개)
    "codex-5.4-mini": "gpt-5.4-mini",
}
# agy(제미니 게이트웨이) 경유 모델 전반 — id → {표시명 base, 지원 effort}.
# effort는 agy 표시명 접미사 "(Low/Medium/High)"로 선택된다(모델명에 내장).
# 등록 금지: agy에 Claude Sonnet/Opus (Thinking)도 있으나 네이티브 claude -p 경로가
#   있는 모델을 다른 게이트웨이로 중복 등록하면 같은 모델이 다른 조건으로 위장된다
#   (측정 무결성).
GEMINI_MODELS = {
    "gemini-3-pro": {"name": "Gemini 3.1 Pro", "efforts": ("low", "high")},
    "gemini-3.5-flash": {"name": "Gemini 3.5 Flash", "efforts": ("low", "medium", "high")},
    "gpt-oss-120b": {"name": "GPT-OSS 120B", "efforts": ("medium",)},  # agy 경유 오픈소스
}
# vendor CLI 격리용 빈 작업 디렉터리(레포 파일/AGENTS.md 오염 차단)
ISO_DIR = "/tmp/llm-bench-iso"
AGY_BIN = os.path.expanduser("~/.local/bin/agy")

# 사람이 읽기 좋은 짧은 별칭(리포트 라벨용)
MODEL_ALIASES = {
    "claude-opus-4-0": "4.0",
    "claude-opus-4-1": "4.1",
    "claude-opus-4-5": "4.5",
    "claude-opus-4-6": "4.6",
    "claude-opus-4-7": "4.7",
    "claude-opus-4-8": "4.8",
    "claude-haiku-4-5": "h4.5",
    "claude-sonnet-4-6": "s4.6",
    "claude-sonnet-5": "s5",
    "claude-fable-5": "f5",
    "codex-5.5": "cx5.5",
    "codex-5.4": "cx5.4",
    "codex-5.6-luna": "5.6 Luna",
    "codex-5.6-sol": "5.6 Sol",
    "codex-5.6-terra": "5.6 Terra",
    "codex-5.4-mini": "cx5.4m",
    "gemini-3-pro": "G3pro",
    "gemini-3.5-flash": "G3.5 Flash",
    "gpt-oss-120b": "OSS 120B",
}

# 풀 표시명(관전 화면 레인·범례·표용) — 축약 별칭은 좁은 지면 전용으로 유지.
# codex-*의 실제 모델은 GPT 계열이므로 표시명도 GPT로(라우팅명과 구분).
MODEL_NAMES = {
    "claude-opus-4-0": "Claude Opus 4.0",
    "claude-opus-4-1": "Claude Opus 4.1",
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-fable-5": "Claude Fable 5",
    "codex-5.5": "GPT-5.5",
    "codex-5.4": "GPT-5.4",
    "codex-5.6-luna": "GPT-5.6 Luna",
    "codex-5.6-sol": "GPT-5.6 Sol",
    "codex-5.6-terra": "GPT-5.6 Terra",
    "codex-5.4-mini": "GPT-5.4 Mini",
    "gemini-3-pro": "Gemini 3.1 Pro",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gpt-oss-120b": "GPT-OSS 120B",
}


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


# --- Mindmatch 게임 파일럿 모델 -----------------------------------------
# 꼬맨틀 스위트 기본 대상(R6): 저비용 클로드 + 코덱스 루나, effort low.
GAME_PILOT_MODELS = ["claude-haiku-4-5", "codex-5.6-luna"]


def vendor(model: str) -> str:
    if model in CODEX_MODELS:
        return "codex"
    if model in GEMINI_MODELS:
        return "gemini"
    return "claude"


def model_efforts(model: str) -> tuple[str, ...]:
    """모델이 실제 지원하는 effort 단계 — UI·엔진이 공유하는 단일 소스.

    미지원 단계를 노출하면 같은 조건이 다른 레인으로 위장된다(측정 무결성):
    codex CLI엔 max가 없고(xhigh가 상한), agy 게이트웨이 모델은 표시명 접미사로
    선택 가능한 단계만 지원한다(모델별로 다름 — 실호출 검증 기준).
    """
    v = vendor(model)
    if v == "gemini":
        return tuple(GEMINI_MODELS[model]["efforts"])
    if v == "codex":
        return ("low", "medium", "high", "xhigh")
    return ("low", "medium", "high", "xhigh", "max")

# --- 호출 고정 파라미터 -------------------------------------------------
# 두 모델을 "같은 게이트웨이/같은 조건"으로 통과시키는 게 공정성의 핵심.
# effort를 고정하지 않으면 추론량 차이가 능력차로 둔갑한다.
DEFAULT_EFFORT = "high"            # low < medium < high < xhigh < max
DEFAULT_REPEATS = 1               # 축별로 N회 반복(분산 추정용). 비용 고려 기본 1.
CALL_TIMEOUT_S = 600

# 격리: 벤치 대상이 유저의 CLAUDE.md/프로젝트 설정/도구를 못 보게 한다.
# (유저 글로벌 지침이 새면 "성격" 측정이 오염된다 → setting-sources 비움)
BENCH_SYSTEM_PROMPT = (
    "You are the subject of a controlled benchmark. "
    "Respond to the user's message directly and completely. "
    "Do not use any tools. Do not ask for clarification unless the task is "
    "genuinely impossible to attempt; instead make a reasonable assumption and answer."
)
# print 모드에서 노출되는 기본 도구 전부 차단(순수 텍스트 능력만 측정)
DISALLOWED_TOOLS = [
    "Bash", "Read", "Edit", "Write", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "NotebookEdit", "TodoWrite",
]


def alias(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def ensure_dirs() -> None:
    for d in (RESULTS, RAW, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    Path(ISO_DIR).mkdir(parents=True, exist_ok=True)


def env_guard() -> None:
    """ANTHROPIC_API_KEY가 잡혀 있으면 구독이 아니라 종량제 API로 과금된다.

    이 벤치는 구독 기반 `claude -p` 경로를 전제로 한다(메모리 기준).
    의도치 않은 과금을 막기 위해 경고만 띄운다(차단하진 않음).
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        import sys
        print(
            "[warn] ANTHROPIC_API_KEY가 설정돼 있음 → 구독이 아니라 API 종량제로 과금될 수 있음.",
            file=sys.stderr,
        )
