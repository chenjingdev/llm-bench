"""서브프로세스 타임아웃 회귀 테스트 — 손자 프로세스를 그룹째 죽이는지 검증.

실런 arena-20260715-183011에서 24레인 중 13개 스레드가 read()에 영구 정지한 버그의
재현/방지. 원인: subprocess.run(cmd, capture_output=True, timeout=T)은 타임아웃 시
직접 자식만 kill한다. codex/claude/gemini CLI가 스폰한 손자(node 서버 등)가 stdout/stderr
파이프의 write end를 물고 살아있으면:
  - Python 3.12+ 에선 run()의 타임아웃 처리가 파이프를 EOF까지 read → **영원히 블록**.
  - Python 3.9/3.14 에선 run()이 wait()만 하고 반환하지만 **손자를 고아로 방치**
    (파이프 계속 보유 → 다음 배치에서 fd/자원 누적).
수정: subprocess.Popen(start_new_session=True) + 타임아웃 시 os.killpg(SIGKILL)로 그룹째
kill → 손자까지 죽어 파이프가 닫히고 유한 시간에 반환.

핵심 단언은 "타임아웃 뒤 손자가 살아남지 않는다"이다(= killpg가 실제로 동작). 이 단언은
Python 버전과 무관하게 수정 유무를 가른다(수정 전엔 hang 또는 고아 잔존, 수정 후엔 소멸).
'유한 시간 반환' 단언도 병행하되, 테스트 인터프리터가 3.9면 수정 전에도 빨리 반환하므로
그것만으로는 회귀를 못 잡는다 — 손자 소멸 단언이 진짜 가드다.

실제 CLI에 의존하지 않고, os.fork로 손자가 파이프를 확실히 물게 하는 가짜 명령으로 재현한다.
스레드+join(8) 하드 데드라인을 걸어, 수정이 깨져도(hang 변종) suite가 행하지 않게 한다.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

from bench import client, config

# 직접 자식(파이썬)이 손자를 fork하고, 부모·손자 모두 fd1(stdout 파이프)을 물고 오래 산다.
# lsof로 확인: 부모/손자가 동일 파이프 write end를 공유. 직접 자식만 kill하면 손자가
# 파이프를 놓지 않아 read가 손자 종료까지 블록/고아 잔존. 그룹째 kill해야 손자까지 죽는다.
# 셸 백그라운드 잡의 fd 처리는 플랫폼/타이밍 의존이라 재현이 불안정하므로 os.fork로 결정화.
_GRANDCHILD_SRC = (
    "import os, time\n"
    "if os.fork() == 0:\n"
    "    time.sleep(30)  # 손자: fd1(파이프 write end)을 물고 대기\n"
    "else:\n"
    "    time.sleep(30)  # 부모(직접 자식): 대기\n"
)


def _make_grandchild_script(tmp_path) -> str:
    """fd1을 물고 있는 손자를 결정적으로 스폰하는 실행 스크립트 경로 반환(고유 경로)."""
    p = tmp_path / "grandchild.py"
    p.write_text(f"#!{sys.executable}\n{_GRANDCHILD_SRC}")
    p.chmod(0o755)
    return str(p)


def _alive(script: str) -> list[str]:
    """이 스크립트를 실행 중인 프로세스 pid 목록(부모/손자). 고유 경로라 오탐 없음."""
    out = subprocess.run(["pgrep", "-f", script], capture_output=True, text=True).stdout
    return [p for p in out.split() if p]


def _wait_all_dead(script: str, timeout: float = 3.0) -> bool:
    """script 관련 프로세스가 모두 사라질 때까지 폴링. 살아남으면 False(회귀)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(script):
            return True
        time.sleep(0.1)
    return not _alive(script)


def _run_with_deadline(fn, deadline_s: float = 8.0):
    """fn()을 daemon 스레드에서 실행하고 deadline까지 대기. (box, hung) 반환.

    hung=True면 수정이 깨져 read가 블록된 것(hang 변종). box에는 result/exc/elapsed.
    """
    box: dict = {}

    def go():
        t0 = time.monotonic()
        try:
            box["result"] = fn()
        except BaseException as e:  # 예외도 "유한 시간에 반환"으로 간주
            box["exc"] = e
        box["elapsed"] = time.monotonic() - t0

    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(deadline_s)
    return box, th.is_alive()


def test_run_helper_kills_process_group_on_timeout(tmp_path):
    """_run은 손자가 파이프를 물고 있어도 유한 시간에 TimeoutExpired로 반환하고,
    손자 프로세스를 그룹째 죽인다(고아 잔존/hang 없음)."""
    script = _make_grandchild_script(tmp_path)
    try:
        box, hung = _run_with_deadline(lambda: client._run([script], timeout=2))

        assert not hung, "_run이 행 — 손자가 파이프를 잡아 read가 블록됨(수정 전 hang 변종)"
        assert isinstance(box.get("exc"), subprocess.TimeoutExpired), (
            f"TimeoutExpired 기대, 실제: result={box.get('result')!r} exc={box.get('exc')!r}"
        )
        assert box["elapsed"] < 5, f"반환이 너무 느림: {box['elapsed']:.1f}s"
        # 진짜 가드: 그룹째 kill 되었으므로 손자가 살아남지 않는다.
        assert _wait_all_dead(script), (
            "타임아웃 뒤 손자 프로세스가 살아있음 — killpg 미동작(수정 전 고아 잔존 동작)"
        )
    finally:
        subprocess.run(["pkill", "-9", "-f", script], capture_output=True)


def test_call_gemini_timeout_returns_fast(tmp_path, monkeypatch):
    """공개 경로(_call_gemini)가 timeout 시 ok=False/error='timeout'을 신속 반환하고
    손자를 그룹째 정리한다."""
    script = _make_grandchild_script(tmp_path)
    monkeypatch.setattr(config, "AGY_BIN", script)
    monkeypatch.setattr(config, "ISO_DIR", str(tmp_path / "iso"))
    monkeypatch.setattr(config, "GEMINI_MODELS", {"fake": {"name": "Fake"}})
    try:
        box, hung = _run_with_deadline(
            lambda: client._call_gemini("fake", "hi", "low", timeout=2)
        )

        assert not hung, "_call_gemini가 행 — 손자가 파이프를 잡아 read가 블록됨(수정 전 동작)"
        if "exc" in box:
            raise box["exc"]
        res = box["result"]
        assert res.ok is False
        assert res.error == "timeout"
        assert box["elapsed"] < 5, f"반환이 너무 느림: {box['elapsed']:.1f}s"
        assert _wait_all_dead(script), (
            "타임아웃 뒤 손자 프로세스가 살아있음 — killpg 미동작(수정 전 동작)"
        )
    finally:
        subprocess.run(["pkill", "-9", "-f", script], capture_output=True)
