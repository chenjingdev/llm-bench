"""Korean Semantle-style game with a versioned local embedding oracle."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import threading
from dataclasses import dataclass

from .. import embed
from .base import Action, GameState


TARGET_WORDS = (
    "가족", "거울", "겨울", "경찰", "고양이", "공원", "공장", "과학", "교실", "구름",
    "기억", "기차", "나무", "날씨", "노래", "농장", "눈물", "도서관", "도시", "동물",
    "마음", "바다", "박물관", "병원", "봄날", "비행기", "사랑", "사진", "산책", "선생님",
    "소설", "시장", "신문", "약속", "여행", "영화", "우산", "우체국", "운동", "음악",
    "의사", "자동차", "자전거", "정원", "지하철", "친구", "학교", "행복", "호수", "회사",
)

REFERENCE_WORDS = (
    "가게", "가구", "가방", "가을", "간호사", "강물", "거리", "건물", "게임", "결혼",
    "경기", "경제", "계단", "계절", "고기", "고향", "공기", "공부", "공연", "공책",
    "공항", "과일", "관계", "광장", "교사", "교육", "교통", "구두", "구조", "그림",
    "글씨", "기계", "기분", "기술", "기억", "기차", "길거리", "꽃밭", "나라", "낚시",
    "남편", "냉장고", "노동", "노인", "농부", "다리", "대학", "도로", "독서", "동생",
    "마을", "마차", "문학", "문화", "물고기", "미술", "바람", "바위", "반지", "방송",
    "배우", "버스", "병원", "부모", "부엌", "비밀", "사무실", "사슴", "사전", "산업",
    "새벽", "생각", "생활", "서점", "선물", "선박", "설탕", "세계", "소리", "손님",
    "수업", "숲길", "시간", "식당", "아기", "아버지", "아침", "어머니", "언어", "얼굴",
    "역사", "연구", "연극", "열차", "예술", "요리", "운전", "웃음", "은행", "음식",
    "의자", "이야기", "인간", "일기", "작가", "작품", "전쟁", "전화", "정치", "종이",
    "주방", "주택", "직업", "책상", "청소", "축구", "커피", "컴퓨터", "태양", "편지",
    "평화", "하늘", "학생", "항구", "형제", "회의", "휴대폰", "희망",
) + TARGET_WORDS

# 'GUESS'로 시작하는 줄과 그 뒤 인자를 분리 캡처한다(형식 오류 원인 구분용).
_GUESS_LINE = re.compile(r"^[ \t]*GUESS\b[ \t]*(.*)$", re.MULTILINE | re.IGNORECASE)
# 추측 인자: 한 글자 이상 12자 이하 한국어 단어(정답은 모두 2자+라 승부엔 지장 없음).
_WORD = re.compile(r"^[가-힣]{1,12}$")


def _digest_words(words: tuple[str, ...]) -> str:
    payload = json.dumps(words, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass
class SimilarityFeedback:
    similarity: float
    rank: int


# 아레나 오라클 임베딩 모델 — 이 머신 ollama에 keep_alive:-1로 영구 상주(즉답).
# 검증 결과 prefix 없이(prefix=False) 근접어·분포 분리도가 더 좋아 무프리픽스로 쓴다.
# embed.py의 기본 MODEL(nomic)은 creativity 등 다른 서브시스템용이라 건드리지 않는다.
ORACLE_MODEL = "qwen3-embedding:8b"


class EmbeddingOracle:
    """Exact cosine rank over a fixed reference vocabulary."""

    def __init__(self, words: tuple[str, ...] = REFERENCE_WORDS,
                 model: str = ORACLE_MODEL):
        self.words = tuple(dict.fromkeys(words))
        self.model = model
        self.model_identity = embed.model_info(model)
        self.vocab_digest = _digest_words(self.words)
        self._lock = threading.Lock()
        self._vectors = embed.embed(list(self.words), prefix=False, model=self.model)
        self._index = {word: i for i, word in enumerate(self.words)}

    @property
    def metadata(self) -> dict:
        return {
            "type": "exact-cosine-reference-rank",
            "embedding_model": self.model,
            "embedding_digest": self.model_identity.get("digest", "unknown"),
            "reference_words": len(self.words),
            "vocab_digest": self.vocab_digest,
            "rank_scope": "pinned-reference-vocabulary",
        }

    def prepare(self, target: str) -> dict:
        target_vec = self._vector(target)
        scores = [embed.cosine(target_vec, v) for v in self._vectors]
        return {"target": target, "target_vec": target_vec, "scores": scores}

    def _vector(self, word: str) -> list[float]:
        idx = self._index.get(word)
        if idx is not None:
            return self._vectors[idx]
        with self._lock:
            return embed.embed([word], prefix=False, model=self.model)[0]

    def evaluate(self, prepared: dict, guess: str) -> SimilarityFeedback:
        if guess == prepared["target"]:
            return SimilarityFeedback(1.0, 1)
        score = embed.cosine(prepared["target_vec"], self._vector(guess))
        rank = 1 + sum(1 for value in prepared["scores"] if value > score)
        return SimilarityFeedback(score, rank)

    def pair_cosine(self, a: str, b: str) -> float:
        """두 추측 단어의 코사인(오라클 벡터). sim_to_prev 계산용, 결정론적."""
        return embed.cosine(self._vector(a), self._vector(b))


class KoreanSemantle:
    id = "ko-semantle"
    version = "1.3.0"
    DEFAULT_MAX_TURNS = 40

    def __init__(self, oracle: EmbeddingOracle | None = None, max_turns: int = DEFAULT_MAX_TURNS):
        self.oracle = oracle or EmbeddingOracle()
        self.max_turns = max_turns

    @property
    def metadata(self) -> dict:
        return {"game": self.id, "version": self.version, **self.oracle.metadata}

    def reset(self, seed: int) -> GameState:
        target = random.Random(seed).choice(TARGET_WORDS)
        state = GameState(self.id, self.version, seed, self.max_turns, target)
        state.private["oracle"] = self.oracle.prepare(target)
        return state

    @staticmethod
    def _pct_label(rank: int, n: int) -> str:
        """순위 → '상위 P%' 백분위 라벨(순위 절대값보다 해석 가능한 신호)."""
        pct = max(1, round(100 * rank / n)) if n else 100
        return f"상위 {pct}%"

    def render(self, state: GameState) -> str:
        n = len(self.oracle.words)
        rows = []
        for event in state.history:
            if event.get("valid"):
                rank = event["rank"]
                rows.append(
                    f'{event["turn"]}. {event["guess"]} — '
                    f'유사도 {event["similarity"] * 100:.2f} / {n}개 중 {rank}위 '
                    f'({self._pct_label(rank, n)})'
                )
            else:
                rows.append(f'{event["turn"]}. 형식 오류 — {event.get("error", "invalid")}')
        history = "\n".join(rows) if rows else "아직 추측 없음"
        best = min((e["rank"] for e in state.history if e.get("valid")), default=None)
        best_line = (f"지금까지 최고: {best}위 ({self._pct_label(best, n)})"
                     if best else "지금까지 최고: 없음")
        # 프롬프트 캐시 정렬: [고정 규칙] → [이전 기록(append-only, 오래된 것부터)]
        # → [변동부: 현재 턴/최고/출력 지시]. 규칙+기록은 턴이 지나도 바이트가 연장만
        # 되므로 계정 내 prefix 캐시가 히트한다. 변동부를 뒤로 몰아 초반 prefix가 깨지지
        # 않게 한다(측정 조건 변경 → version 범프).
        return (
            "비밀 한국어 단어를 찾는 의미 추측 게임입니다.\n"
            "유사도는 고정된 로컬 임베딩으로 계산하며, 높을수록 정답과 의미가 가깝습니다.\n"
            "유사도 절대값은 캘리브레이션이 어렵습니다. 순위와 백분위를 더 신뢰하세요.\n"
            f"순위는 고정 비교 어휘 {n}개 안에서의 참고 순위입니다(낮을수록 정답에 가까움).\n"
            "매 응답에는 정확히 한 개의 행동만 포함하세요.\n\n"
            f"이전 기록:\n{history}\n\n"
            f"현재 턴: {state.turn + 1}/{state.max_turns}\n"
            f"{best_line}\n\n"
            "다음 형식으로 한 글자 이상의 한국어 단어 하나만 추측하세요.\n"
            "GUESS <단어>"
        )

    def parse(self, text: str) -> Action:
        raw = text or ""
        lines = _GUESS_LINE.findall(raw)
        if len(lines) != 1:
            # GUESS 줄이 0개 또는 2개 이상 — 실제 원인을 말한다.
            return Action("guess", "", raw, False,
                          "정확히 한 개의 'GUESS <단어>' 줄이 필요합니다")
        arg = lines[0].strip()
        if not _WORD.match(arg):
            # GUESS 줄은 있으나 인자가 한국어 단어 형식이 아니다.
            return Action("guess", "", raw, False,
                          "GUESS 뒤에는 한 글자 이상의 한국어 단어 하나만 쓰세요")
        return Action("guess", arg, raw)

    def step(self, state: GameState, action: Action) -> dict:
        state.turn += 1
        if not action.valid:
            event = {"turn": state.turn, "valid": False, "raw": action.raw,
                     "error": action.error}
        elif action.value in state.seen:
            event = {"turn": state.turn, "valid": False, "raw": action.raw,
                     "guess": action.value, "error": "duplicate guess"}
        else:
            # 직전 유효 추측(있으면)과의 코사인 — 고착 진단용. 첫 유효 추측은 null.
            prev = next((e["guess"] for e in reversed(state.history)
                         if e.get("valid") and "guess" in e), None)
            state.seen.add(action.value)
            feedback = self.oracle.evaluate(state.private["oracle"], action.value)
            sim_to_prev = (round(self.oracle.pair_cosine(prev, action.value), 8)
                           if prev is not None else None)
            event = {
                "turn": state.turn,
                "valid": True,
                "raw": action.raw,
                "guess": action.value,
                "similarity": round(feedback.similarity, 8),
                "rank": feedback.rank,
                "sim_to_prev": sim_to_prev,
            }
            if action.value == state.secret:
                state.solved = True
                state.done = True
                state.stop_reason = "solved"
        state.history.append(event)
        if state.turn >= state.max_turns and not state.done:
            state.done = True
            state.stop_reason = "max_turns"
        return event

    def result(self, state: GameState) -> dict:
        valid = [event for event in state.history if event.get("valid")]
        best_rank = min((event["rank"] for event in valid), default=None)
        curve = []
        best = len(self.oracle.words) + 1
        for event in state.history:
            if event.get("valid"):
                best = min(best, event["rank"])
            curve.append(best)
        auc = None
        if curve:
            denom = math.log(len(self.oracle.words) + 1)
            auc = sum(1.0 - math.log(max(rank, 1)) / denom for rank in curve) / len(curve)
        score = (0.5 + 0.5 * (1 - (state.turn - 1) / state.max_turns)
                 if state.solved else 0.5 * (auc or 0.0))
        # 고착(fixation) 지표(임계 상수 없이):
        #  - max_plateau: best_rank가 개선되지 않은 최장 연속 유효 턴 수
        #  - fixation_sim: 그 정체 구간의 sim_to_prev 중앙값(같은 의미 동네를 맴돌았는지)
        best_seen = None
        streak = 0
        max_plateau = 0
        plateau_sims: list[float] = []
        for event in valid:
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
        fixation_sim = round(statistics.median(plateau_sims), 8) if plateau_sims else None
        return {
            "solved": state.solved,
            "turns": state.turn,
            "best_rank": best_rank,
            "best_rank_curve": curve,
            "best_log_rank_auc": round(auc, 6) if auc is not None else None,
            "valid_guesses": len(valid),
            "invalid_actions": len(state.history) - len(valid),
            "score": round(score, 6),
            "max_plateau": max_plateau,
            "fixation_sim": fixation_sim,
            "stop_reason": state.stop_reason,
        }
