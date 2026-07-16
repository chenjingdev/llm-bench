"""ko-minefield — 의미 지뢰밭.

꼬맨틀(ko-semantle)의 변형: 정답 단어를 의미 추측으로 찾되, REFERENCE_WORDS에서
결정론으로 뽑힌 숨은 '지뢰' 의미 영역 2곳을 피해야 한다. 지뢰에 근접하면 목숨을
잃고(폭발) 그 턴의 정답 피드백을 몰수당한다. 목숨을 모두 잃으면 패배.

semantle.py에서 EmbeddingOracle / TARGET_WORDS / REFERENCE_WORDS만 재사용한다
(semantle.py는 다른 워커 소유 — 수정하지 않는다). best_rank_curve / max_plateau /
fixation_sim / score는 semantle과 동일한 정의를 이 모듈에 자체 구현한다(코드 복제
허용 — semantle 내부 함수에 의존하지 않는다).
"""

from __future__ import annotations

import math
import random
import re
import statistics

from .base import Action, GameState, instance_tag_line, new_nonce
from .semantle import EmbeddingOracle, REFERENCE_WORDS, TARGET_WORDS


# 게임 상수(metadata에 노출 — 재생 검증 identity 대상).
LIVES = 3
MINES = 2
BOOM_RANK = 3
WARN_RANK = 15
# 지뢰 적격 기준: 정답 기준 프레임에서 지뢰의 순위가 이 값을 초과해야 한다(정답에
# 정직하게 접근하는 경로 위에 지뢰가 놓이지 않게 한다). metadata에는 넣지 않는다
# (version에 내재된 규칙). reset 결정론에만 관여.
MINE_RANK_MIN = 30

DEFAULT_MAX_TURNS = 40

# 파싱은 semantle과 동일 형식 `GUESS <단어>`. 코드 복제 허용 — 자체 정규식을 둔다.
_GUESS_LINE = re.compile(r"^[ \t]*GUESS\b[ \t]*(.*)$", re.MULTILINE | re.IGNORECASE)
_WORD = re.compile(r"^[가-힣]{1,12}$")


class KoreanMinefield:
    id = "ko-minefield"
    version = "1.1.0"   # 프롬프트에 에피소드 인스턴스 태그 추가(측정 조건 변경)
    DEFAULT_MAX_TURNS = DEFAULT_MAX_TURNS

    # --- 프로토콜 속성(계약 §1/§5) ---
    TURN_FIELDS = ("guess", "similarity", "rank", "sim_to_prev", "mine_event", "lives")
    INVALID_KEEP = ("guess",)
    LIVE_LAST_FIELDS = ("guess", "similarity", "rank")
    RESULT_FIELDS = ("solved", "turns", "score", "best_rank", "best_rank_curve",
                     "lives_left", "booms", "warns", "mines", "max_plateau",
                     "fixation_sim")
    needs_ollama = True
    # 재생 검증 허용 오차: 임베딩 서버의 동시 요청 코배칭 노이즈(실측 Δsim ~2e-4,
    # Δrank ≤1)를 계기 오차로 수용. mine_event/lives/booms 등 판정 결과는 정확 비교
    # 유지 — 노이즈로 폭발/경보 경계가 뒤집힌 재생 불능 판은 그대로 드러나야 한다.
    TOLERANT_FIELDS = {"similarity": 2e-3, "sim_to_prev": 2e-3, "rank": 2,
                       "best_rank": 2, "best_rank_curve": 2, "score": 5e-3,
                       "max_plateau": 2, "fixation_sim": 2e-3}

    def __init__(self, oracle: EmbeddingOracle | None = None,
                 max_turns: int = DEFAULT_MAX_TURNS):
        self.oracle = oracle or EmbeddingOracle()
        self.max_turns = max_turns

    @property
    def metadata(self) -> dict:
        # 상수 + 오라클 identity. verify_run이 manifest.oracle와 대조한다.
        return {
            "game": self.id,
            "version": self.version,
            "lives": LIVES,
            "mines": MINES,
            "boom_rank": BOOM_RANK,
            "warn_rank": WARN_RANK,
            **self.oracle.metadata,
        }

    # ------------------------------------------------------------------
    # reset — 정답 + 지뢰 추첨(완전 결정론: random.Random(seed)에서만 파생)
    # ------------------------------------------------------------------
    def reset(self, seed: int, nonce: str | None = None) -> GameState:
        rng = random.Random(seed)
        target = rng.choice(TARGET_WORDS)
        state = GameState(self.id, self.version, seed, self.max_turns, target)
        state.nonce = new_nonce() if nonce is None else nonce  # 에피소드 시작 시 1회 발급

        prepared_target = self.oracle.prepare(target)
        mine_words: list[str] = []
        prepared_mines: list[dict] = []
        # 제약(전부 결정론, 같은 rng 순차 재추첨): 지뢰 ≠ 정답, 지뢰끼리 서로 다름,
        # 정답 프레임에서 각 지뢰의 순위 > MINE_RANK_MIN(정답에서 충분히 멀다).
        attempts = 0
        while len(mine_words) < MINES:
            attempts += 1
            if attempts > 100000:  # 안전밸브 — 정상 어휘에선 도달하지 않는다.
                raise RuntimeError("지뢰 추첨 실패: 제약을 만족하는 지뢰를 찾지 못함")
            mine = rng.choice(REFERENCE_WORDS)
            if mine == target or mine in mine_words:
                continue
            if self.oracle.evaluate(prepared_target, mine).rank <= MINE_RANK_MIN:
                continue
            mine_words.append(mine)
            prepared_mines.append(self.oracle.prepare(mine))

        state.private["oracle"] = prepared_target
        state.private["mines"] = prepared_mines
        state.private["mine_words"] = mine_words
        state.private["lives"] = LIVES
        return state

    # ------------------------------------------------------------------
    # render — [고정 규칙] → [이전 기록(append-only)] → [변동부] (프롬프트 캐시 정렬)
    # ------------------------------------------------------------------
    @staticmethod
    def _pct_label(rank: int, n: int) -> str:
        pct = max(1, round(100 * rank / n)) if n else 100
        return f"상위 {pct}%"

    def render(self, state: GameState) -> str:
        n = len(self.oracle.words)
        rows = []
        for event in state.history:
            if not event.get("valid"):
                rows.append(f'{event["turn"]}. 형식 오류 — {event.get("error", "invalid")}')
            elif event.get("mine_event") == "boom":
                # 폭발 턴: 유사도/순위 몰수 → 순위 대신 폭발만 표기.
                rows.append(
                    f'{event["turn"]}. {event["guess"]} — '
                    f'지뢰 폭발! (남은 목숨 {event["lives"]})')
            else:
                rank = event["rank"]
                warn = " [지뢰 접근 경보]" if event.get("mine_event") == "warn" else ""
                rows.append(
                    f'{event["turn"]}. {event["guess"]} — '
                    f'유사도 {event["similarity"] * 100:.2f} / {n}개 중 {rank}위 '
                    f'({self._pct_label(rank, n)}){warn}')
        history = "\n".join(rows) if rows else "아직 추측 없음"

        best = min((e["rank"] for e in state.history
                    if e.get("valid") and "rank" in e), default=None)
        best_line = (f"지금까지 최고: {best}위 ({self._pct_label(best, n)})"
                     if best else "지금까지 최고: 없음")
        lives = state.private.get("lives", LIVES)

        # 맨 앞 인스턴스 태그(에피소드 내 불변)는 레인별 프롬프트 비동일화용 고정부다.
        return (
            instance_tag_line(state) +
            "숨은 한국어 단어를 찾는 의미 추측 게임입니다.\n"
            "유사도는 고정된 로컬 임베딩으로 계산하며, 높을수록 정답과 의미가 가깝습니다.\n"
            "유사도 절대값은 캘리브레이션이 어렵습니다. 순위와 백분위를 더 신뢰하세요.\n"
            f"순위는 고정 비교 어휘 {n}개 안에서의 참고 순위입니다(낮을수록 정답에 가까움).\n"
            f"이 세계에는 숨은 지뢰 의미 영역 {MINES}곳이 있고, 시작 목숨은 {LIVES}개입니다.\n"
            f"추측이 어느 지뢰든 근접 상위 {BOOM_RANK}위 이내면 지뢰 폭발 — 목숨을 잃고 "
            "그 턴의 정답 유사도·순위 피드백은 몰수됩니다.\n"
            f"어느 지뢰든 상위 {WARN_RANK}위 이내면 지뢰 접근 경보만 표시됩니다(피드백은 정상).\n"
            "목숨을 모두 잃으면 패배합니다.\n"
            "매 응답에는 정확히 한 개의 행동만 포함하세요.\n\n"
            f"이전 기록:\n{history}\n\n"
            f"현재 턴: {state.turn + 1}/{state.max_turns}\n"
            f"남은 목숨: {lives}/{LIVES}\n"
            f"{best_line}\n\n"
            "다음 형식으로 한 글자 이상의 한국어 단어 하나만 추측하세요.\n"
            "GUESS <단어>"
        )

    # ------------------------------------------------------------------
    # parse — semantle과 동일 형식
    # ------------------------------------------------------------------
    def parse(self, text: str) -> Action:
        raw = text or ""
        lines = _GUESS_LINE.findall(raw)
        if len(lines) != 1:
            return Action("guess", "", raw, False,
                          "정확히 한 개의 'GUESS <단어>' 줄이 필요합니다")
        arg = lines[0].strip()
        if not _WORD.match(arg):
            return Action("guess", "", raw, False,
                          "GUESS 뒤에는 한 글자 이상의 한국어 단어 하나만 쓰세요")
        return Action("guess", arg, raw)

    # ------------------------------------------------------------------
    # step — 판정 순서: 승리 → 폭발 → 경보 → 정상
    # ------------------------------------------------------------------
    def step(self, state: GameState, action: Action) -> dict:
        state.turn += 1
        if not action.valid:
            event = {"turn": state.turn, "valid": False, "raw": action.raw,
                     "error": action.error}
        elif action.value in state.seen:
            event = {"turn": state.turn, "valid": False, "raw": action.raw,
                     "guess": action.value, "error": "duplicate guess"}
        else:
            # 직전 유효 추측(폭발 턴 추측도 '유효 추측'으로 연쇄에 포함)과의 코사인.
            prev = next((e["guess"] for e in reversed(state.history)
                         if e.get("valid") and "guess" in e), None)
            state.seen.add(action.value)
            guess = action.value
            lives = state.private["lives"]

            if guess == state.secret:
                # (1) 승리 — 지뢰 무시.
                feedback = self.oracle.evaluate(state.private["oracle"], guess)
                event = {
                    "turn": state.turn, "valid": True, "raw": action.raw,
                    "guess": guess,
                    "similarity": round(feedback.similarity, 8),
                    "rank": feedback.rank,
                    "sim_to_prev": self._sim_to_prev(prev, guess),
                    "mine_event": None,
                    "lives": lives,
                }
                state.solved = True
                state.done = True
                state.stop_reason = "solved"
            else:
                mine_rank = min(self.oracle.evaluate(pm, guess).rank
                                for pm in state.private["mines"])
                if mine_rank <= BOOM_RANK:
                    # (2) 폭발 — 목숨 -1, 이 턴의 정답 유사도/순위/sim_to_prev 몰수.
                    state.private["lives"] -= 1
                    lives = state.private["lives"]
                    event = {"turn": state.turn, "valid": True, "raw": action.raw,
                             "guess": guess, "mine_event": "boom", "lives": lives}
                    if lives <= 0:
                        state.done = True
                        state.stop_reason = "mined"
                else:
                    # (3) 경보(≤WARN_RANK) 또는 (4) 정상 — 둘 다 정상 피드백.
                    feedback = self.oracle.evaluate(state.private["oracle"], guess)
                    event = {
                        "turn": state.turn, "valid": True, "raw": action.raw,
                        "guess": guess,
                        "similarity": round(feedback.similarity, 8),
                        "rank": feedback.rank,
                        "sim_to_prev": self._sim_to_prev(prev, guess),
                        "mine_event": "warn" if mine_rank <= WARN_RANK else None,
                        "lives": lives,
                    }

        state.history.append(event)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return event

    def _sim_to_prev(self, prev: str | None, guess: str):
        if prev is None:
            return None
        return round(self.oracle.pair_cosine(prev, guess), 8)

    # ------------------------------------------------------------------
    # progress — 매 턴 레코드/live.json에 병합될 누적 지표
    # ------------------------------------------------------------------
    def progress(self, state: GameState) -> dict:
        ranks = [e["rank"] for e in state.history if e.get("valid") and "rank" in e]
        return {"best_rank": min(ranks) if ranks else None,
                "lives": state.private.get("lives", LIVES)}

    # ------------------------------------------------------------------
    # result — semantle과 동일 정의의 지표 + 지뢰 지표(자체 구현)
    # ------------------------------------------------------------------
    def result(self, state: GameState) -> dict:
        n = len(self.oracle.words)
        # 순위 있는 유효 턴만(폭발 턴은 rank 없음).
        ranked = [e for e in state.history if e.get("valid") and "rank" in e]
        best_rank = min((e["rank"] for e in ranked), default=None)

        # best_rank_curve: 히스토리 길이만큼, 폭발/무효 턴은 직전 best 유지.
        curve = []
        best = n + 1
        for event in state.history:
            if event.get("valid") and "rank" in event:
                best = min(best, event["rank"])
            curve.append(best)

        auc = None
        if curve:
            denom = math.log(n + 1)
            auc = sum(1.0 - math.log(max(rank, 1)) / denom for rank in curve) / len(curve)
        score = (0.5 + 0.5 * (1 - (state.turn - 1) / state.max_turns)
                 if state.solved else 0.5 * (auc or 0.0))
        if state.stop_reason == "mined":
            score = 0.0

        # 고착 지표(semantle과 동일 정의, 순위 있는 유효 턴 대상):
        best_seen = None
        streak = 0
        max_plateau = 0
        plateau_sims: list[float] = []
        for event in ranked:
            rank = event["rank"]
            if best_seen is None or rank < best_seen:
                best_seen = rank
                streak = 0
            else:
                streak += 1
                max_plateau = max(max_plateau, streak)
                sim = event.get("sim_to_prev")
                if sim is not None:
                    plateau_sims.append(sim)
        fixation_sim = (round(statistics.median(plateau_sims), 8)
                        if plateau_sims else None)

        booms = sum(1 for e in state.history if e.get("mine_event") == "boom")
        warns = sum(1 for e in state.history if e.get("mine_event") == "warn")
        valid = [e for e in state.history if e.get("valid")]

        return {
            "solved": state.solved,
            "turns": state.turn,
            "score": round(score, 6),
            "best_rank": best_rank,
            "best_rank_curve": curve,
            "lives_left": state.private.get("lives", LIVES),
            "booms": booms,
            "warns": warns,
            "mines": list(state.private.get("mine_words", [])),  # episode_end에서만 공개
            "max_plateau": max_plateau,
            "fixation_sim": fixation_sim,
            # 계약 RESULT_FIELDS 밖 확장(semantle 관례 — 통합 편의, episode_end엔 미복사):
            "stop_reason": state.stop_reason,
            "invalid_actions": len(state.history) - len(valid),
        }

    # ------------------------------------------------------------------
    # summary_stats — summary.json 병합용 게임별 집계(빈 리스트 안전)
    # ------------------------------------------------------------------
    def summary_stats(self, episode_ends: list[dict]) -> dict:
        best_ranks = [e["best_rank"] for e in episode_ends
                      if e.get("best_rank") is not None]
        booms = [e.get("booms", 0) for e in episode_ends]
        mined = [e for e in episode_ends if e.get("lives_left") == 0]
        return {
            "median_best_rank": statistics.median(best_ranks) if best_ranks else None,
            "median_booms": statistics.median(booms) if booms else None,
            "mined_rate": (round(len(mined) / len(episode_ends), 4)
                           if episode_ends else 0.0),
        }
