"""ko-maze — 숨은 지도 탐험(안개 낀 7×7 perfect maze).

임베딩 불필요(needs_ollama=False). 미로·시작·목표는 전부 random.Random(seed)에서
파생되는 순수 결정론 세계다. 동일 seed로 reset한 뒤 저장된 raw를 재적용하면 이벤트가
바이트 수준으로 동일하게 재현된다(재생 검증 계약).

좌표계: (x, y). x는 동쪽으로 갈수록 커지고 y는 남쪽으로 갈수록 커진다. (0,0)=북서 모서리.
목표까지의 맨해튼 거리(dist)는 이벤트 저장 전용이며 render()에 절대 노출하지 않는다.
"""

from __future__ import annotations

import random
import re
import statistics
from collections import deque

from .base import Action, GameState, instance_tag_line, new_nonce


# 열린 방향/기록 고정 표기 순서: 북·남·동·서 (관찰과 open 문자열이 이 순서를 따른다).
DIRECTIONS = ("북", "남", "동", "서")
# 각 방향의 좌표 델타. 북=y-1, 남=y+1, 동=x+1, 서=x-1.
_DELTA = {"북": (0, -1), "남": (0, 1), "동": (1, 0), "서": (-1, 0)}
_OPPOSITE = {"북": "남", "남": "북", "동": "서", "서": "동"}

# 8방위 방향표: (sign(dx), sign(dy)) → 방위명. dx=gx-x, dy=gy-y.
_BEARING = {
    (0, -1): "북", (1, -1): "북동", (1, 0): "동", (1, 1): "남동",
    (0, 1): "남", (-1, 1): "남서", (-1, 0): "서", (-1, -1): "북서",
}

# 'MOVE'로 시작하는 줄과 그 뒤 인자를 분리 캡처한다(형식 오류 원인 구분용).
_MOVE_LINE = re.compile(r"^[ \t]*MOVE\b[ \t]*(.*)$", re.MULTILINE | re.IGNORECASE)


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _bearing(x: int, y: int, gx: int, gy: int) -> str:
    """현재 칸에서 목표 칸을 향한 8방위. 도착 시 '도착'."""
    dx, dy = gx - x, gy - y
    if dx == 0 and dy == 0:
        return "도착"
    return _BEARING[(_sign(dx), _sign(dy))]


def _generate_maze(rng: random.Random, size: int) -> dict[tuple[int, int], set[str]]:
    """recursive backtracker로 perfect maze(전 칸 연결 신장트리) 생성.

    open_dirs[(x,y)] = 그 칸에서 통로가 열린 방향 집합. 무방향이라 (x,y)의 '동'이 열리면
    (x+1,y)의 '서'도 열린다. 이웃 후보는 DIRECTIONS 고정 순서로 모아 rng.choice로 뽑아
    seed만으로 완전 결정된다.
    """
    open_dirs: dict[tuple[int, int], set[str]] = {
        (x, y): set() for x in range(size) for y in range(size)
    }
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        x, y = stack[-1]
        candidates = []
        for d in DIRECTIONS:
            ddx, ddy = _DELTA[d]
            nx, ny = x + ddx, y + ddy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                candidates.append((d, nx, ny))
        if candidates:
            d, nx, ny = rng.choice(candidates)
            open_dirs[(x, y)].add(d)
            open_dirs[(nx, ny)].add(_OPPOSITE[d])
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    return open_dirs


def _shortest_path_len(open_dirs: dict[tuple[int, int], set[str]],
                       start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    """미로 통로를 따른 start→goal 최단 경로 길이(간선 수). 도달 불가면 None."""
    if start == goal:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (x, y), d = q.popleft()
        for direction in DIRECTIONS:
            if direction in open_dirs[(x, y)]:
                ddx, ddy = _DELTA[direction]
                nxt = (x + ddx, y + ddy)
                if nxt not in seen:
                    if nxt == goal:
                        return d + 1
                    seen.add(nxt)
                    q.append((nxt, d + 1))
    return None


def _pick_start_goal(rng: random.Random, open_dirs, size):
    """같은 rng로 시작·목표를 순차 재추첨. 맨해튼 ≥ 6, 최단 경로 ≤ 28을 만족할 때까지.

    perfect maze는 항상 연결이라 경로는 존재한다. 재추첨은 seed만으로 결정된다.
    """
    cells = [(x, y) for y in range(size) for x in range(size)]
    while True:
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if abs(goal[0] - start[0]) + abs(goal[1] - start[1]) < 6:
            continue
        path_len = _shortest_path_len(open_dirs, start, goal)
        if path_len is None or path_len > 28:
            continue
        return start, goal, path_len


class KoreanMaze:
    id = "ko-maze"
    version = "1.1.0"   # 프롬프트에 에피소드 인스턴스 태그 추가(측정 조건 변경)
    DEFAULT_MAX_TURNS = 40
    SIZE = 7

    # 프로토콜 속성(base.Game 확장) — arena.py가 저장·재생 검증에 사용.
    TURN_FIELDS = ("move", "ok", "pos", "open", "bearing", "dist")
    INVALID_KEEP = ()
    LIVE_LAST_FIELDS = ("move", "pos", "bearing")
    RESULT_FIELDS = ("solved", "turns", "score", "moves", "bumps", "revisits",
                     "explored_ratio", "path_efficiency", "min_dist", "dist_curve")
    needs_ollama = False

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self.max_turns = max_turns

    @property
    def metadata(self) -> dict:
        return {"game": self.id, "version": self.version,
                "type": "deterministic-maze", "size": self.SIZE}

    def reset(self, seed: int, nonce: str | None = None) -> GameState:
        rng = random.Random(seed)
        open_dirs = _generate_maze(rng, self.SIZE)
        start, goal, path_len = _pick_start_goal(rng, open_dirs, self.SIZE)
        gx, gy = goal
        state = GameState(self.id, self.version, seed, self.max_turns, f"{gx},{gy}")
        state.nonce = new_nonce() if nonce is None else nonce  # 에피소드 시작 시 1회 발급
        start_dist = abs(gx - start[0]) + abs(gy - start[1])
        # 미로·시작·목표·진행 카운터는 전부 private에 둔다(정답 비누출).
        state.private = {
            "open_dirs": open_dirs,
            "size": self.SIZE,
            "start": start,
            "goal": goal,
            "path_len": path_len,
            "pos": [start[0], start[1]],
            "visited": {start},
            "bumps": 0,
            "moves": 0,
            "successful_moves": 0,
            "revisits": 0,
            "min_dist": start_dist,
            "dist_curve": [],
        }
        return state

    # ------------------------------------------------------------------
    # 관찰 조립 헬퍼
    # ------------------------------------------------------------------
    def _open_str(self, priv: dict, cell: tuple[int, int]) -> str:
        return "·".join(d for d in DIRECTIONS if d in priv["open_dirs"][cell])

    def render(self, state: GameState) -> str:
        priv = state.private
        size = priv["size"]
        gx, gy = priv["goal"]
        sx, sy = priv["start"]
        start_open = self._open_str(priv, (sx, sy))
        start_bearing = _bearing(sx, sy, gx, gy)

        rows = []
        for event in state.history:
            if not event.get("valid"):
                rows.append(f'{event["turn"]}. 형식 오류 — {event.get("error", "invalid")}')
            elif event["ok"]:
                px, py = event["pos"]
                dx, dy = _DELTA[event["move"]]
                fx, fy = px - dx, py - dy       # 이동 전 위치를 pos·move로 복원
                rows.append(
                    f'{event["turn"]}. ({fx},{fy}) {event["move"]} → '
                    f'({px},{py}) 열림:{event["open"]} 방위:{event["bearing"]}'
                )
            else:                               # 벽에 막힘(위치 불변)
                px, py = event["pos"]
                rows.append(f'{event["turn"]}. ({px},{py}) {event["move"]} → 벽에 막힘')
        history = "\n".join(rows) if rows else "아직 이동 없음"

        cx, cy = priv["pos"]
        cur_open = self._open_str(priv, (cx, cy))
        cur_bearing = _bearing(cx, cy, gx, gy)

        # 프롬프트 캐시 정렬: [고정 규칙 + 시작 관찰] → [이동 기록(append-only, 오래된
        # 것부터)] → [변동부: 현재 턴/위치/열림/방위/행동 지시]. 규칙+시작 관찰은
        # 에피소드 내 불변, 기록은 연장만 되므로 prefix 캐시가 히트한다.
        # 맨 앞 인스턴스 태그(에피소드 내 불변)는 레인별 프롬프트 비동일화용 고정부다.
        return (
            instance_tag_line(state) +
            "안개 낀 미로를 탐험해 숨은 목표 칸을 찾는 게임입니다.\n"
            f"미로는 {size}×{size} 격자이며 모든 칸이 통로로 연결되어 있습니다.\n"
            "좌표는 (x, y)로 나타내며, x는 동쪽으로 갈수록 커지고 y는 남쪽으로 갈수록 "
            "커집니다. (0,0)은 북서쪽 모서리입니다.\n"
            "관찰로는 현재 위치, 지금 칸에서 열린 방향, 목표의 방위만 주어집니다. "
            "목표까지의 거리는 알려주지 않습니다.\n"
            "방위는 북/북동/동/남동/남/남서/서/북서 중 하나이며, 목표 칸에 서면 '도착'입니다.\n"
            "매 응답에는 정확히 한 개의 행동만 포함하세요.\n\n"
            f"시작 위치: ({sx},{sy})\n"
            f"시작 관찰: 열림:{start_open} 방위:{start_bearing}\n\n"
            f"이동 기록:\n{history}\n\n"
            f"현재 턴: {state.turn + 1}/{state.max_turns}\n"
            f"현재 위치: ({cx},{cy})\n"
            f"현재 열린 방향: {cur_open}\n"
            f"현재 목표 방위: {cur_bearing}\n\n"
            "다음 형식으로 한 방향만 이동하세요.\n"
            "MOVE <북|남|동|서>"
        )

    def parse(self, text: str) -> Action:
        raw = text or ""
        lines = _MOVE_LINE.findall(raw)
        if len(lines) != 1:
            # MOVE 줄이 0개 또는 2개 이상 — 실제 원인을 말한다.
            return Action("move", "", raw, False,
                          "정확히 한 개의 'MOVE <북|남|동|서>' 줄이 필요합니다")
        arg = lines[0].strip()
        if arg not in _DELTA:
            # MOVE 줄은 있으나 인자가 네 방향 중 하나가 아니다.
            return Action("move", "", raw, False,
                          "MOVE 뒤에는 북/남/동/서 중 하나만 쓰세요")
        return Action("move", arg, raw)

    def step(self, state: GameState, action: Action) -> dict:
        state.turn += 1
        priv = state.private
        gx, gy = priv["goal"]
        if not action.valid:
            event = {"turn": state.turn, "valid": False, "raw": action.raw,
                     "error": action.error}
        else:
            move = action.value
            x, y = priv["pos"]
            ddx, ddy = _DELTA[move]
            priv["moves"] += 1
            if move in priv["open_dirs"][(x, y)]:
                # 이동 성공 — 새 칸으로 전진.
                nx, ny = x + ddx, y + ddy
                priv["pos"] = [nx, ny]
                priv["successful_moves"] += 1
                if (nx, ny) in priv["visited"]:
                    priv["revisits"] += 1
                priv["visited"].add((nx, ny))
                ok, cx, cy = True, nx, ny
            else:
                # 벽에 막힘 — 위치 불변, 턴은 소모.
                priv["bumps"] += 1
                ok, cx, cy = False, x, y
            dist = abs(gx - cx) + abs(gy - cy)
            event = {
                "turn": state.turn,
                "valid": True,
                "raw": action.raw,
                "move": move,
                "ok": ok,
                "pos": [cx, cy],
                "open": self._open_str(priv, (cx, cy)),
                "bearing": _bearing(cx, cy, gx, gy),
                "dist": dist,           # 저장 전용 — render 비노출
            }
            if ok and (cx, cy) == (gx, gy):
                state.solved = True
                state.done = True
                state.stop_reason = "solved"
        # dist_curve/min_dist는 유효·무효 무관 매 턴 현재 거리를 기록한다(progress와 일치).
        px, py = priv["pos"]
        cur_dist = abs(gx - px) + abs(gy - py)
        priv["dist_curve"].append(cur_dist)
        priv["min_dist"] = min(priv["min_dist"], cur_dist)
        state.history.append(event)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return event

    def progress(self, state: GameState) -> dict:
        priv = state.private
        if "goal" not in priv:
            # 0턴 실패 마감 경로: arena._finalize_live가 reset 전 빈 상태로도 호출한다.
            return {"dist": None, "explored": 0.0, "bumps": 0}
        gx, gy = priv["goal"]
        x, y = priv["pos"]
        return {
            "dist": abs(gx - x) + abs(gy - y),
            "explored": round(len(priv["visited"]) / (priv["size"] ** 2), 4),
            "bumps": priv["bumps"],
        }

    def result(self, state: GameState) -> dict:
        priv = state.private
        solved = state.solved
        successful = priv["successful_moves"]
        path_efficiency = (round(priv["path_len"] / successful, 6)
                           if solved and successful else None)
        score = (round((state.max_turns - state.turn + 1) / state.max_turns, 6)
                 if solved else 0.0)
        return {
            "solved": solved,
            "turns": state.turn,
            "score": score,
            "moves": priv["moves"],
            "bumps": priv["bumps"],
            "revisits": priv["revisits"],
            "explored_ratio": round(len(priv["visited"]) / (priv["size"] ** 2), 4),
            "path_efficiency": path_efficiency,
            "min_dist": priv["min_dist"],
            "dist_curve": list(priv["dist_curve"]),
            "stop_reason": state.stop_reason,
        }

    def summary_stats(self, episode_ends: list[dict]) -> dict:
        min_dists = [e["min_dist"] for e in episode_ends if e.get("min_dist") is not None]
        exploreds = [e["explored_ratio"] for e in episode_ends
                     if e.get("explored_ratio") is not None]
        return {
            "median_min_dist": (round(statistics.median(min_dists), 4)
                                if min_dists else None),
            "median_explored": (round(statistics.median(exploreds), 4)
                                if exploreds else None),
        }
