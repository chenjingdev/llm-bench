"""ko-maze 유닛 테스트 — 임베딩/네트워크/실모델 없이 순수 결정론으로 돈다.

미로·시작·목표는 seed만으로 완전히 결정되므로, 동일 seed reset 후 저장된 raw를
재적용하면 이벤트가 바이트 수준으로 동일해야 한다(재생 검증 계약).
"""

import os
from collections import deque

from bench.games.maze import (
    DIRECTIONS, _DELTA, _OPPOSITE, _bearing, _shortest_path_len, KoreanMaze,
)


# ----------------------------------------------------------------------
# 테스트 헬퍼 — 미로 내부를 들여다보며 결정론적으로 목표까지 항해한다.
# ----------------------------------------------------------------------
def _flood(open_dirs, start, size):
    """통로를 따라 start에서 도달 가능한 칸 집합(연결성 확인용)."""
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for d in DIRECTIONS:
            if d in open_dirs[(x, y)]:
                dx, dy = _DELTA[d]
                nb = (x + dx, y + dy)
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
    return seen


def _path_dirs(open_dirs, start, goal):
    """start→goal 최단 경로를 방향 문자열 리스트로(BFS 복원)."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for d in DIRECTIONS:
            if d in open_dirs[cur]:
                dx, dy = _DELTA[d]
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt not in prev:
                    prev[nxt] = (cur, d)
                    q.append(nxt)
    dirs = []
    cur = goal
    while prev[cur] is not None:
        pcell, d = prev[cur]
        dirs.append(d)
        cur = pcell
    dirs.reverse()
    return dirs


def _seed_with_start_wall():
    """시작 칸에 벽(닫힌 방향)이 하나라도 있는 seed와 상태를 찾는다."""
    g = KoreanMaze()
    for seed in range(200):
        s = g.reset(seed)
        start = tuple(s.private["start"])
        if any(d not in s.private["open_dirs"][start] for d in DIRECTIONS):
            return g, seed, s
    raise AssertionError("모든 seed의 시작 칸이 사방 개방 — 있을 수 없음")


# ----------------------------------------------------------------------
# 프로토콜 속성 계약
# ----------------------------------------------------------------------
def test_class_contract_attributes():
    g = KoreanMaze()
    assert g.id == "ko-maze"
    assert g.version == "1.1.0"
    assert g.DEFAULT_MAX_TURNS == 40
    assert g.needs_ollama is False
    assert KoreanMaze.TURN_FIELDS == ("move", "ok", "pos", "open", "bearing", "dist")
    assert KoreanMaze.INVALID_KEEP == ()
    assert KoreanMaze.LIVE_LAST_FIELDS == ("move", "pos", "bearing")
    assert KoreanMaze.RESULT_FIELDS == (
        "solved", "turns", "score", "moves", "bumps", "revisits",
        "explored_ratio", "path_efficiency", "min_dist", "dist_curve")
    # result()가 RESULT_FIELDS를 전부 담는다.
    res = g.result(g.reset(1))
    for f in KoreanMaze.RESULT_FIELDS:
        assert f in res
    # progress 키는 계약대로.
    assert set(g.progress(g.reset(1))) == {"dist", "explored", "bumps"}
    # 기본 생성자 max_turns 반영.
    assert KoreanMaze(max_turns=12).max_turns == 12


def test_metadata_has_no_secret():
    md = KoreanMaze().metadata
    assert md["game"] == "ko-maze"
    assert md["version"] == "1.1.0"
    assert md["size"] == 7
    # 목표 좌표류는 metadata에 없다.
    assert "goal" not in md and "secret" not in md


# ----------------------------------------------------------------------
# 방위·좌표계
# ----------------------------------------------------------------------
def test_bearing_all_eight_directions():
    # y는 남쪽으로 증가, x는 동쪽으로 증가 — 8방위가 좌표계와 일치.
    assert _bearing(3, 3, 3, 0) == "북"
    assert _bearing(3, 3, 6, 0) == "북동"
    assert _bearing(3, 3, 6, 3) == "동"
    assert _bearing(3, 3, 6, 6) == "남동"
    assert _bearing(3, 3, 3, 6) == "남"
    assert _bearing(3, 3, 0, 6) == "남서"
    assert _bearing(3, 3, 0, 3) == "서"
    assert _bearing(3, 3, 0, 0) == "북서"
    assert _bearing(3, 3, 3, 3) == "도착"


# ----------------------------------------------------------------------
# 결정론: 같은 seed → 같은 미로·시작·목표
# ----------------------------------------------------------------------
def test_seed_determinism_same_world():
    g = KoreanMaze()
    a, b = g.reset(7), g.reset(7)
    assert a.secret == b.secret
    assert a.private["start"] == b.private["start"]
    assert a.private["goal"] == b.private["goal"]
    assert a.private["open_dirs"] == b.private["open_dirs"]
    assert a.private["path_len"] == b.private["path_len"]
    # 서로 다른 KoreanMaze 인스턴스도 동일 seed면 동일 세계.
    assert KoreanMaze().reset(123).secret == KoreanMaze().reset(123).secret


def test_secret_matches_goal():
    g = KoreanMaze()
    for seed in (0, 1, 5, 42, 100):
        s = g.reset(seed)
        gx, gy = s.private["goal"]
        assert s.secret == f"{gx},{gy}"


# ----------------------------------------------------------------------
# 미로 연결성·제약 충족(perfect maze = 신장트리)
# ----------------------------------------------------------------------
def test_maze_connectivity_and_constraints():
    g = KoreanMaze()
    for seed in range(300):
        s = g.reset(seed)
        priv = s.private
        open_dirs, size = priv["open_dirs"], priv["size"]
        # 무방향 일관성: (x,y)의 d 열림 ⟺ 이웃의 반대 방향 열림, 경계 밖 통로 없음.
        for (x, y), dirs in open_dirs.items():
            for d in dirs:
                dx, dy = _DELTA[d]
                nb = (x + dx, y + dy)
                assert 0 <= nb[0] < size and 0 <= nb[1] < size
                assert _OPPOSITE[d] in open_dirs[nb]
        # 신장트리: 간선 수 = 칸 수 - 1.
        edges = sum(len(v) for v in open_dirs.values()) // 2
        assert edges == size * size - 1
        # 전 칸 도달 가능(연결).
        assert len(_flood(open_dirs, tuple(priv["start"]), size)) == size * size
        # 시작·목표 제약: 맨해튼 ≥ 6, 최단 경로 ≤ 28.
        sx, sy = priv["start"]
        gx, gy = priv["goal"]
        assert abs(gx - sx) + abs(gy - sy) >= 6
        assert priv["path_len"] <= 28
        # path_len은 실제 최단 경로와 일치.
        assert priv["path_len"] == _shortest_path_len(open_dirs, priv["start"], priv["goal"])


# ----------------------------------------------------------------------
# 이동/벽/무효/도달 각 경로
# ----------------------------------------------------------------------
def test_successful_move_updates_position_and_event():
    g = KoreanMaze()
    s = g.reset(1)
    priv = s.private
    start = tuple(priv["start"])
    d = next(dd for dd in DIRECTIONS if dd in priv["open_dirs"][start])
    ev = g.step(s, g.parse(f"MOVE {d}"))
    dx, dy = _DELTA[d]
    nxt = [start[0] + dx, start[1] + dy]
    assert ev["valid"] is True and ev["ok"] is True
    assert ev["move"] == d
    assert ev["pos"] == nxt
    assert priv["pos"] == nxt
    assert priv["moves"] == 1 and priv["successful_moves"] == 1 and priv["bumps"] == 0
    # 성공 이벤트는 TURN_FIELDS를 전부 담는다.
    for f in KoreanMaze.TURN_FIELDS:
        assert f in ev
    # dist는 이동 후 맨해튼 거리.
    gx, gy = priv["goal"]
    assert ev["dist"] == abs(gx - nxt[0]) + abs(gy - nxt[1])


def test_wall_bump_keeps_position_and_counts():
    g, seed, s = _seed_with_start_wall()
    priv = s.private
    start = tuple(priv["start"])
    closed = [d for d in DIRECTIONS if d not in priv["open_dirs"][start]]
    ev = g.step(s, g.parse(f"MOVE {closed[0]}"))
    assert ev["valid"] is True and ev["ok"] is False
    assert ev["pos"] == [start[0], start[1]]          # 위치 불변
    assert priv["pos"] == [start[0], start[1]]
    assert priv["bumps"] == 1                          # 벽 충돌 누적
    assert priv["moves"] == 1                          # 유효 이동 시도로 집계
    assert priv["successful_moves"] == 0
    assert s.turn == 1                                 # 턴은 소모
    # 벽 이벤트도 pos/open/bearing/dist(불변 값)를 담는다.
    for f in KoreanMaze.TURN_FIELDS:
        assert f in ev


def test_invalid_actions_and_error_messages():
    g = KoreanMaze()
    assert g.parse("아무 방향도 없습니다").valid is False
    assert g.parse("MOVE 북\nMOVE 남").valid is False       # 두 줄
    bad = g.parse("MOVE 위")                                # 방향 아님
    none_line = g.parse("설명만 합니다")                    # MOVE 줄 0개
    assert bad.valid is False and none_line.valid is False
    assert "북/남/동/서" in bad.error
    assert "한 개" in none_line.error
    assert bad.error != none_line.error                     # 원인 구분
    ok = g.parse("먼저 생각합니다.\nMOVE 동")
    assert ok.valid is True and ok.value == "동"


def test_invalid_step_consumes_turn_without_moving():
    g = KoreanMaze(max_turns=5)
    s = g.reset(1)
    start = list(s.private["start"])
    ev = g.step(s, g.parse("엉뚱한 행동"))
    assert ev["valid"] is False
    assert "error" in ev and "move" not in ev and "dist" not in ev
    assert s.turn == 1
    assert s.private["moves"] == 0                          # 무효는 이동 시도 아님
    assert s.private["pos"] == start
    # 무효 턴도 dist_curve에는 현재 거리를 남긴다.
    assert len(s.private["dist_curve"]) == 1


def test_reaching_goal_solves_and_stops():
    g = KoreanMaze(max_turns=40)
    s = g.reset(2)
    priv = s.private
    dirs = _path_dirs(priv["open_dirs"], tuple(priv["start"]), tuple(priv["goal"]))
    events = [g.step(s, g.parse(f"MOVE {d}")) for d in dirs]
    assert s.solved is True and s.done is True
    assert s.stop_reason == "solved"
    last = events[-1]
    assert last["ok"] is True and last["bearing"] == "도착" and last["dist"] == 0
    assert last["pos"] == list(priv["goal"])
    # 도달 후 더 이상 진행하지 않음(마지막 이벤트에서 종료).
    assert len(events) == len(dirs)
    res = g.result(s)
    assert res["solved"] is True
    assert res["min_dist"] == 0
    assert res["path_efficiency"] == 1.0                    # 최단 경로 항해
    assert res["score"] == round((40 - len(dirs) + 1) / 40, 6)


def test_max_turns_exhaustion_sets_done():
    g = KoreanMaze(max_turns=3)
    s = g.reset(5)
    for _ in range(3):
        g.step(s, g.parse("무효 행동"))
    assert s.done is True and s.solved is False
    assert s.stop_reason == "max_turns"
    res = g.result(s)
    assert res["solved"] is False
    assert res["score"] == 0.0
    assert res["path_efficiency"] is None
    assert res["turns"] == 3
    assert len(res["dist_curve"]) == 3


# ----------------------------------------------------------------------
# 지표 계산: revisits·bumps·explored_ratio·path_efficiency·dist_curve
# ----------------------------------------------------------------------
def test_revisits_counted_on_backtrack():
    g = KoreanMaze()
    s = g.reset(1)
    priv = s.private
    start = tuple(priv["start"])
    d = next(dd for dd in DIRECTIONS if dd in priv["open_dirs"][start])
    g.step(s, g.parse(f"MOVE {d}"))                # 이웃으로 전진
    g.step(s, g.parse(f"MOVE {_OPPOSITE[d]}"))     # 되돌아옴(통로는 무방향)
    assert tuple(priv["pos"]) == start
    assert priv["revisits"] == 1                   # 시작 칸 재진입
    assert priv["successful_moves"] == 2
    # 방문 고유 칸은 2개(시작·이웃) → explored = 2/49.
    assert g.progress(s)["explored"] == round(2 / 49, 4)


def test_bumps_tracked_in_progress_and_result():
    g, seed, s = _seed_with_start_wall()
    priv = s.private
    start = tuple(priv["start"])
    closed = [d for d in DIRECTIONS if d not in priv["open_dirs"][start]]
    g.step(s, g.parse(f"MOVE {closed[0]}"))
    assert g.progress(s)["bumps"] == 1
    res = g.result(s)
    assert res["bumps"] == 1
    assert res["moves"] == 1
    assert res["revisits"] == 0
    assert res["path_efficiency"] is None          # 미해결
    assert res["explored_ratio"] == round(1 / 49, 4)   # 시작 칸만 방문


def test_dist_curve_and_progress_track_every_turn():
    g = KoreanMaze(max_turns=6)
    s = g.reset(4)
    g.step(s, g.parse("MOVE 북"))                  # 성공이든 벽이든 dist 기록
    g.step(s, g.parse("무효"))                     # 무효도 dist 기록
    curve = s.private["dist_curve"]
    assert len(curve) == s.turn == 2
    prog = g.progress(s)
    gx, gy = s.private["goal"]
    px, py = s.private["pos"]
    assert prog["dist"] == abs(gx - px) + abs(gy - py)
    assert curve[-1] == prog["dist"]               # 마지막 곡선값 = 현재 거리
    assert s.private["min_dist"] == min([abs(gx - s.private["start"][0])
                                         + abs(gy - s.private["start"][1])] + curve)


def test_path_efficiency_below_one_when_wandering():
    g = KoreanMaze(max_turns=40)
    s = g.reset(2)
    priv = s.private
    start = tuple(priv["start"])
    # 시작에서 왕복 2수(낭비) 후 최단 경로로 목표까지 — 성공 이동 수가 path_len보다 큼.
    d0 = next(dd for dd in DIRECTIONS if dd in priv["open_dirs"][start])
    g.step(s, g.parse(f"MOVE {d0}"))
    g.step(s, g.parse(f"MOVE {_OPPOSITE[d0]}"))
    dirs = _path_dirs(priv["open_dirs"], start, tuple(priv["goal"]))
    for d in dirs:
        g.step(s, g.parse(f"MOVE {d}"))
    res = g.result(s)
    assert res["solved"] is True
    assert res["moves"] == 2 + len(dirs)
    assert res["path_efficiency"] == round(priv["path_len"] / (2 + len(dirs)), 6)
    assert res["path_efficiency"] < 1.0


# ----------------------------------------------------------------------
# 재생 재현성: 저장된 raw 재적용 → 이벤트 동일
# ----------------------------------------------------------------------
def test_replay_determinism_arbitrary_sequence():
    g = KoreanMaze(max_turns=40)
    raws = ["MOVE 북", "MOVE 남", "MOVE 동", "생각\nMOVE 서",
            "형식 위반", "MOVE 위", "MOVE 북"]

    def run():
        s = g.reset(3)
        return [g.step(s, g.parse(r)) for r in raws]

    a, b = run(), run()
    assert a == b                                  # 재생 대조 대상 전부 일치


def test_replay_determinism_full_solve_and_result():
    g = KoreanMaze(max_turns=40)

    def run():
        s = g.reset(11)
        dirs = _path_dirs(s.private["open_dirs"], tuple(s.private["start"]),
                          tuple(s.private["goal"]))
        events = [g.step(s, g.parse(f"MOVE {d}")) for d in dirs]
        return events, g.result(s)

    (ev_a, res_a), (ev_b, res_b) = run(), run()
    assert ev_a == ev_b
    assert res_a == res_b
    assert res_a["solved"] is True


# ----------------------------------------------------------------------
# 정답·거리 비노출
# ----------------------------------------------------------------------
def test_render_hides_goal_and_distance():
    g = KoreanMaze()
    s = g.reset(2)
    priv = s.private
    dirs = _path_dirs(priv["open_dirs"], tuple(priv["start"]), tuple(priv["goal"]))
    for d in dirs[:2]:                             # 목표 미도달 상태로 두 수
        g.step(s, g.parse(f"MOVE {d}"))
    text = g.render(s)
    # 목표 좌표(secret)는 어디에도 없다(좌표는 전부 한 자리라 부분일치 위험 없음).
    assert s.secret not in text
    # 마지막 유효 이동 기록 줄은 방위로 끝나고 dist가 붙지 않는다.
    ev = next(e for e in reversed(s.history) if e.get("valid"))
    px, py = ev["pos"]
    dx, dy = _DELTA[ev["move"]]
    fx, fy = px - dx, py - dy
    row = (f'{ev["turn"]}. ({fx},{fy}) {ev["move"]} → '
           f'({px},{py}) 열림:{ev["open"]} 방위:{ev["bearing"]}')
    assert (row + "\n") in text                    # 줄이 방위에서 끝남(뒤에 dist 없음)
    assert f'{row} {ev["dist"]}' not in text
    assert "거리:" not in text                     # 거리 라벨 자체가 없음


def test_render_shows_rules_start_and_current_observation():
    g = KoreanMaze()
    s = g.reset(2)
    priv = s.private
    sx, sy = priv["start"]
    text = g.render(s)
    assert "7×7" in text
    assert "매 응답에는 정확히 한 개의 행동만 포함하세요." in text
    assert f"시작 위치: ({sx},{sy})" in text
    assert "시작 관찰: 열림:" in text
    assert f"현재 위치: ({sx},{sy})" in text        # 첫 턴 현재 위치 = 시작
    assert "현재 목표 방위:" in text
    assert "MOVE <북|남|동|서>" in text


# ----------------------------------------------------------------------
# 프롬프트 캐시 정렬: 턴 진행 시 render prefix가 연장만 되는지
# ----------------------------------------------------------------------
def test_render_prefix_stable_for_cache_alignment():
    g = KoreanMaze(max_turns=20)
    s = g.reset(2)
    dirs = _path_dirs(s.private["open_dirs"], tuple(s.private["start"]),
                      tuple(s.private["goal"]))
    g.step(s, g.parse(f"MOVE {dirs[0]}"))
    p_k = g.render(s)                              # 기록 1행
    g.step(s, g.parse(f"MOVE {dirs[1]}"))
    p_k1 = g.render(s)                             # 기록 2행(append-only 연장)

    common = os.path.commonprefix([p_k, p_k1])
    # 고정 규칙 + 시작 관찰 + 기존 기록행이 공통 prefix에 그대로.
    assert "매 응답에는 정확히 한 개의 행동만 포함하세요." in common
    assert "시작 관찰: 열림:" in common
    assert "이동 기록:" in common
    assert "1. " in common                        # 첫 기록행이 prefix에 보존
    # 변동부(현재 턴/위치/방위)는 공통 prefix에 없다(divergence 뒤에만).
    assert "현재 턴:" not in common
    assert "현재 위치:" not in common
    assert "현재 목표 방위:" not in common
    # 각 프롬프트에서 변동부는 공통 prefix 뒤 꼬리에 존재.
    assert "현재 턴:" in p_k[len(common):]
    assert "현재 턴:" in p_k1[len(common):]
    # 순서: 이동 기록이 현재 턴보다 앞.
    assert p_k.index("이동 기록:") < p_k.index("현재 턴:")


# ----------------------------------------------------------------------
# summary_stats 집계(빈 리스트 안전)
# ----------------------------------------------------------------------
def test_summary_stats_median_and_empty_safe():
    g = KoreanMaze()
    ends = [
        {"min_dist": 4, "explored_ratio": 0.5},
        {"min_dist": 2, "explored_ratio": 0.3},
        {"min_dist": 0, "explored_ratio": 0.7},
    ]
    st = g.summary_stats(ends)
    assert st == {"median_min_dist": 2, "median_explored": 0.5}
    assert g.summary_stats([]) == {"median_min_dist": None, "median_explored": None}


def test_progress_safe_on_blank_state():
    """0턴 실패 마감 경로: arena._finalize_live가 reset 전 빈 GameState로 호출한다."""
    from bench.games.base import GameState
    game = KoreanMaze()
    blank = GameState(game.id, game.version, 0, game.max_turns, "")
    assert game.progress(blank) == {"dist": None, "explored": 0.0, "bumps": 0}
