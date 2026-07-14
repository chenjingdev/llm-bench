"""CLI — python -m bench {run|report|score|smoke}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import axes as axis_mod
from . import arena, config, probes, report, runner
from .games import game_names as _game_names

ALL_AXES = list(axis_mod.SCORERS.keys())


def cmd_run(a):
    axes = a.axes or ALL_AXES
    models = a.models or config.DEFAULT_MODELS
    run_dir = runner.run(
        axes, models, effort=a.effort, repeats=a.repeats,
        workers=a.workers, limit=a.limit, seed=a.seed, fmt=a.fmt,
    )
    if not a.no_report:
        report.build(run_dir)


def cmd_report(a):
    run_dir = Path(a.run) if a.run else None
    report.build(run_dir)


def cmd_score(a):
    run_dir = Path(a.run) if a.run else report.latest_run()
    scored = report.score_run(run_dir)
    out = {}
    for axis, per in scored["scores"].items():
        out[axis] = {config.alias(m): {"score": r.score, "n": r.n, **r.subscores}
                     for m, r in per.items()}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_arena(a):
    """Mindmatch — 꼬맨틀 게임 실행/검증/대시보드."""
    if a.arena_cmd == "run":
        # --models 항목은 "model" 또는 "model@effort". @ 없으면 --effort를 쓴다.
        participants = a.models or config.GAME_PILOT_MODELS
        run_dir = arena.run_arena(
            a.game, participants, episodes=a.episodes, max_turns=a.max_turns,
            effort=a.effort, seed_base=a.seed, call_timeout=a.call_timeout,
            workers=a.workers,
        )
        print(str(run_dir))
    elif a.arena_cmd == "verify":
        result = arena.verify_run(Path(a.run_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif a.arena_cmd == "serve":
        # 지연 임포트: arena_web는 Worker B가 병렬 작성 중이므로 실행 시점에만 로드
        from . import arena_web
        arena_web.serve(port=a.port, open_browser=not a.no_open)


def cmd_smoke(a):
    """최소 비용 스모크: haiku로 축당 probe 1개씩만."""
    run_dir = runner.run(
        a.axes or ALL_AXES, ["claude-haiku-4-5"],
        effort="low", repeats=1, workers=2, limit=1, seed=a.seed,
    )
    report.build(run_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench", description="llm-bench N-of-1 모델 비교")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="벤치 실행")
    r.add_argument("--axes", nargs="*", choices=ALL_AXES, help=f"기본: {ALL_AXES}")
    r.add_argument("--models", nargs="*", help=f"기본: {config.DEFAULT_MODELS}")
    r.add_argument("--effort", default=config.DEFAULT_EFFORT)
    r.add_argument("--repeats", type=int, default=config.DEFAULT_REPEATS)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--limit", type=int, default=None, help="축당 probe 수 제한(스모크용)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--fmt", default=None, choices=list(probes.FMT_CONTRACTS),
                   help="형식 계약(표면 스타일 상수화): report")
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="최신/지정 run으로 레이더 리포트 생성")
    rp.add_argument("--run", help="results/raw/<run_id> 경로(기본: 최신)")
    rp.set_defaults(func=cmd_report)

    sc = sub.add_parser("score", help="점수만 JSON으로 출력")
    sc.add_argument("--run")
    sc.set_defaults(func=cmd_score)

    sm = sub.add_parser("smoke", help="haiku 최소 호출 스모크 테스트")
    sm.add_argument("--axes", nargs="*", choices=ALL_AXES)
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(func=cmd_smoke)

    ar = sub.add_parser("arena", help="Mindmatch 게임 실행/검증/대시보드")
    arsub = ar.add_subparsers(dest="arena_cmd", required=True)

    arr = arsub.add_parser("run", help="참가자(모델×effort) 동시 플레이")
    arr.add_argument("--game", default="ko-semantle", choices=_game_names())
    arr.add_argument("--models", nargs="*",
                     help="참가자: 'model' 또는 'model@effort' (예: haiku@low haiku@high). "
                          f"기본: {config.GAME_PILOT_MODELS}")
    arr.add_argument("--episodes", type=int, default=3)
    arr.add_argument("--max-turns", type=int, default=None, dest="max_turns")
    arr.add_argument("--effort", default="low",
                     help="@effort 없는 참가자에 적용할 기본 effort")
    arr.add_argument("--seed", type=int, default=None,
                     help="seed_base(기본: 시간). 에피소드 i의 seed=seed_base+i")
    arr.add_argument("--call-timeout", type=int, default=180, dest="call_timeout")
    arr.add_argument("--workers", type=int, default=None,
                     help=f"동시 실행 상한(기본: min(참가자수, {arena.MAX_WORKERS_DEFAULT}))")
    arr.set_defaults(func=cmd_arena)

    arv = arsub.add_parser("verify", help="저장된 run 재생 검증")
    arv.add_argument("run_dir", help="results/arena/<run_id> 경로")
    arv.set_defaults(func=cmd_arena)

    ars = arsub.add_parser("serve", help="Mindmatch 대시보드 서버")
    ars.add_argument("--port", type=int, default=8777)
    ars.add_argument("--no-open", action="store_true", dest="no_open")
    ars.set_defaults(func=cmd_arena)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
