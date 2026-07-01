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

# --- 교차 vendor 모델(코덱스·제미니) -----------------------------------
# 내부 id → 실제 CLI 모델명. 호출 경로는 client.py가 id 접두사로 라우팅.
#   codex-*  → `codex exec -m <gpt-x>` (ChatGPT 구독). 5.3은 계정 미지원이라 제외.
#   gemini-* → `agy -p --model "<표시명>"` (effort가 모델명에 내장 → High 사용).
CODEX_MODELS = {
    "codex-5.5": "gpt-5.5",
    "codex-5.4": "gpt-5.4",
}
GEMINI_MODELS = {
    "gemini-3-pro": "Gemini 3.1 Pro (High)",
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
    "codex-5.5": "cx5.5",
    "codex-5.4": "cx5.4",
    "gemini-3-pro": "G3pro",
}


def vendor(model: str) -> str:
    if model in CODEX_MODELS:
        return "codex"
    if model in GEMINI_MODELS:
        return "gemini"
    return "claude"

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
