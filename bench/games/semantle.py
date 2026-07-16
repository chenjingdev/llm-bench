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
from .base import Action, GameState, new_time_ns


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

# 출력 JSON의 추측 키. 모델은 이 키 하나를 가진 JSON 오브젝트만 뱉어야 한다.
_OUTPUT_KEY = "발화할 단어"
# 시작_기록 선정 밴드 크기 — 유사도 최하위 K개(정답에서 가장 먼 꼴찌권)에서 1개를 뽑는다.
# 레인마다 단어는 다르지만 전부 최하위권이라 출발 정보가치가 균등(다양화+공정성 동시).
_START_BAND_K = 20
# 추측 인자: 한 글자 이상 12자 이하 한국어 단어(정답은 모두 2자+라 승부엔 지장 없음).
_WORD = re.compile(r"^[가-힣]{1,12}$")


def _extract_json_objects(text: str) -> list[str]:
    """텍스트에서 최상위 {...} 오브젝트 문자열들을 순서대로 추출.

    문자열 리터럴 안의 중괄호는 무시하므로 코드펜스(```json)·전후 잡담은 중괄호 밖이라
    자연히 건너뛴다. 중첩 오브젝트는 최상위 하나로만 잡고, 배열 [{...},{...}]은 두 개의
    최상위 오브젝트로 잡힌다(파서의 '정확히 1개' 강제와 부합).
    """
    objs: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                objs.append(text[start:i + 1])
    return objs


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
# honcho 태그를 쓰는 이유: 이 머신 ollama는 honcho 임베딩 + gpt-oss 2개가 keep_alive
# Forever로 슬롯을 점유해 제3 모델(qwen3-embedding:8b) 로드 요청이 무한 대기했다(실런
# 16레인 동반 정지 원인). honcho 태그는 qwen3-embedding:8b와 동일 가중치(qwen3 7.6B
# Q4_K_M, 차이는 num_ctx 8192뿐)라 벡터가 동일하고, 이미 상주 중이라 즉답(실측 0.2초).
ORACLE_MODEL = "qwen3-embedding-honcho-8192"


class EmbeddingOracle:
    """Exact cosine rank over a fixed reference vocabulary."""

    def __init__(self, words: tuple[str, ...] = REFERENCE_WORDS,
                 model: str = ORACLE_MODEL):
        self.words = tuple(dict.fromkeys(words))
        self.model = model
        self.model_identity = embed.model_info(model)
        self.vocab_digest = _digest_words(self.words)
        # 어휘 밖(OOV) 단어 벡터 메모이즈 + in-flight 디둡용. HTTP는 이 전역 락 밖에서
        # 돌고, 락은 아래 두 dict 조작만 짧게 보호한다(서로 다른 단어는 서로 안 막힘).
        self._oov_vectors: dict[str, list[float]] = {}
        self._oov_inflight: dict[str, dict] = {}
        self._oov_guard = threading.Lock()
        # 어휘 벡터를 디스크 캐시로 메모이즈(웜 재빌드의 ~27초 재임베딩 제거, 순수
        # 메모이제이션이라 metadata·measurement_key 불변) + 콜드 스타트 흡수. 캐시
        # 미스는 워밍업을 동기로 태워 모델 로딩(수 분)을 흡수하고, 적중은 즉시 반환하며
        # 워밍업을 백그라운드로 돌려 이후 플레이가 콜드 스타트에 걸리지 않게 한다.
        self._vectors = embed.embed_vocab_cached(list(self.words), prefix=False,
                                                 model=self.model)
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
        # 어휘 밖(OOV) 단어: 프로세스 수명 메모이즈 + in-flight 디둡.
        # embed_vocab_cached(어휘 디스크 캐시)와 같은 논리 — 같은 입력→같은 벡터인 순수
        # 메모이제이션이라 measurement 의미 불변(버전 무관). 같은 단어를 여러 레인이 동시에
        # 추측해도(예: 16레인이 "사람") HTTP는 하나만 나가고 나머지는 그 결과를 공유한다.
        # 서로 다른 단어는 서로를 막지 않는다(전역 락은 dict 조작만, HTTP는 락 밖).
        vec = self._oov_vectors.get(word)
        if vec is not None:
            return vec
        with self._oov_guard:
            vec = self._oov_vectors.get(word)          # 락 안 재확인
            if vec is not None:
                return vec
            inflight = self._oov_inflight.get(word)
            leader = inflight is None
            if leader:
                inflight = {"event": threading.Event(), "vec": None, "exc": None}
                self._oov_inflight[word] = inflight
        if not leader:                                  # 리더의 HTTP 완료를 기다려 공유
            inflight["event"].wait()
            if inflight["exc"] is not None:
                raise inflight["exc"]
            return inflight["vec"]
        try:                                            # 리더: 전역 락 밖에서 HTTP(병렬)
            inflight["vec"] = embed.embed([word], prefix=False, model=self.model)[0]
        except Exception as exc:                        # 재시도 소진 등 — 원인을 드러낸다
            inflight["exc"] = RuntimeError(
                f"오라클 임베딩 실패: {exc} — ollama에 {self.model} 미로드 가능성")
        finally:
            with self._oov_guard:
                if inflight["vec"] is not None:
                    self._oov_vectors[word] = inflight["vec"]
                self._oov_inflight.pop(word, None)
            inflight["event"].set()                     # 대기자 깨우기(성공·실패 공통)
        if inflight["exc"] is not None:
            raise inflight["exc"]
        return inflight["vec"]

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
    version = "1.6.0"   # 에피소드별 시작_기록(꼴찌권 랜덤 단어 실채점) 추가(측정 조건 변경)
    DEFAULT_MAX_TURNS = 40

    # 멀티게임 저장/검증 일반화용 계약 필드(계약 v1 §1). 값·포맷은 기존 저장
    # 스키마와 키·값 수준에서 동일하도록 고정한다(추가만, 로직/버전 무변경).
    TURN_FIELDS = ("guess", "similarity", "rank", "sim_to_prev")
    INVALID_KEEP = ("guess",)
    LIVE_LAST_FIELDS = ("guess", "similarity", "rank")
    # 시작_기록은 (seed, nonce)에서 결정론 재도출되므로 verify가 RESULT_FIELDS 대조로
    # 일치 확인한다(단어=문자열, 유사도/순위=어휘 캐시 벡터 기반이라 정확 재현 → 무관용).
    RESULT_FIELDS = ("solved", "turns", "best_rank", "best_rank_curve", "score",
                     "max_plateau", "fixation_sim", "시작_기록")
    needs_ollama = True
    # 임베딩 오라클의 동시 요청 코배칭 노이즈(실측 Δsimilarity ~2e-4 → Δrank ≤ 1)를
    # 재생 검증에서 수용한다. rank 파생 지표(curve/plateau/score)도 노이즈로 rank가 1
    # 흔들리면 소폭 흔들리므로 함께 허용. 위변조·규칙 위반 수준의 차이는 임계를 넘어 잡힌다.
    TOLERANT_FIELDS = {"similarity": 2e-3, "sim_to_prev": 2e-3, "rank": 2,
                       "best_rank": 2, "best_rank_curve": 2, "score": 5e-3,
                       "max_plateau": 2, "fixation_sim": 2e-3}

    def __init__(self, oracle: EmbeddingOracle | None = None, max_turns: int = DEFAULT_MAX_TURNS):
        self.oracle = oracle or EmbeddingOracle()
        self.max_turns = max_turns

    @property
    def metadata(self) -> dict:
        return {"game": self.id, "version": self.version, **self.oracle.metadata}

    def reset(self, seed: int, nonce: str | None = None) -> GameState:
        target = random.Random(seed).choice(TARGET_WORDS)
        state = GameState(self.id, self.version, seed, self.max_turns, target)
        # 에피소드 시작 시각(time_ns 정수) 1회 발급 — JSON 프롬프트의 time 필드이자
        # 감사용 nonce(episode_end 기록)로 통일. 명시 nonce가 정수면 그 값을 time으로 재사용.
        if nonce is None:
            t = new_time_ns()
            state.nonce = str(t)
        else:
            state.nonce = nonce
            t = int(nonce) if str(nonce).lstrip("-").isdigit() else new_time_ns()
        state.private["time_ns"] = t
        prepared = self.oracle.prepare(target)
        state.private["oracle"] = prepared
        # 시작_기록: 꼴찌권(유사도 최하위 K밴드)에서 (seed,nonce) 결정론 선택 → 실제 채점.
        # 레인마다 의미가 다른 텍스트가 시작 기록으로 들어가되 전부 최하위권이라 공정하다.
        # 모델의 추측이 아니므로 state.history(=이전_기록·턴수·최고_순위)엔 넣지 않는다.
        start_word = self._pick_start_word(prepared, seed, state.nonce)
        fb = self.oracle.evaluate(prepared, start_word)
        state.private["start_record"] = {
            "단어": start_word,
            "유사도": round(fb.similarity * 100, 2),
            "순위": fb.rank,
        }
        return state

    def _pick_start_word(self, prepared: dict, seed: int, nonce: str) -> str:
        """유사도 최하위 K밴드에서 (seed,nonce) 결정론 RNG로 단어 하나 선택(정답 제외).

        정답은 유사도 1위라 최하위 밴드에 애초에 없지만 명시적으로도 배제한다. 동점은
        단어 사전순으로 tie-break해 밴드 구성을 완전 결정론으로 만든다. nonce는 이미
        episode_end에 기록되므로 verify가 seed+nonce로 동일하게 재도출할 수 있다.
        """
        target = prepared["target"]
        words = self.oracle.words
        scores = prepared.get("scores")   # 실오라클은 prepare에서 목표-어휘 코사인을 계산
        if scores is not None:
            graded = [(scores[i], words[i]) for i in range(len(words)) if words[i] != target]
        else:                             # Fake/Stub 등 scores 미제공 → evaluate로 도출
            graded = [(self.oracle.evaluate(prepared, w).similarity, w)
                      for w in words if w != target]
        graded.sort(key=lambda sw: (sw[0], sw[1]))   # 유사도 최하위 우선, 동점은 사전순
        band = graded[:_START_BAND_K]
        return random.Random(f"{seed}:{nonce}").choice(band)[1]

    @staticmethod
    def _pct(rank: int, n: int) -> int:
        """순위 → 상위 백분위(정수). 낮을수록 정답에 가깝다."""
        return max(1, round(100 * rank / n)) if n else 100

    def render(self, state: GameState) -> str:
        """프롬프트를 JSON 페이로드로 직렬화.

        time을 무시할 노이즈가 아니라 스키마의 정식 필드로 격상시켜(값만 제시, 어떤 설명·
        지시도 붙이지 않음) 모델이 데이터로 처리하게 만든다 — 이것이 이번 실험의 핵심 변인.
        캐시 정렬: dict 삽입 순서 = 직렬화 순서이므로 [정적 규칙+time(에피소드 내 불변)]
        → [이전_기록(append-only)] → [변동부: 현재_턴/최고_순위] → [출력 지시] 순으로 넣어,
        턴이 지나도 앞부분이 연장만 되게 한다(배열 append 시 직전 원소에 콤마 1자만 추가되어
        divergence가 맨 끝 경계로 밀린다). 측정 조건 변경 → version 범프.
        """
        n = len(self.oracle.words)
        records = []
        for event in state.history:
            if event.get("valid"):
                rank = event["rank"]
                records.append({
                    "턴": event["turn"],
                    "단어": event["guess"],
                    "유사도": round(event["similarity"] * 100, 2),
                    "순위": rank,
                    "상위백분위": self._pct(rank, n),
                })
            else:
                records.append({"턴": event["turn"],
                                "형식오류": event.get("error", "invalid")})
        best = min((e["rank"] for e in state.history if e.get("valid")), default=None)

        payload = {
            "게임": "비밀 한국어 단어를 찾는 의미 추측 게임",
            "규칙": [
                "유사도는 고정된 로컬 임베딩으로 계산하며, 높을수록 정답과 의미가 가깝습니다.",
                "유사도 절대값은 캘리브레이션이 어렵습니다. 순위와 상위백분위를 더 신뢰하세요.",
                f"순위는 고정 비교 어휘 {n}개 안에서의 참고 순위입니다(낮을수록 정답에 가까움).",
                "시작_기록은 무작위로 제공된 단어 하나의 실제 채점 결과입니다(정답 힌트 아님, 턴을 소모하지 않음).",
                "매 응답에는 정확히 한 개의 추측만 담으세요.",
            ],
            "총_비교_어휘_수": n,
            "time": state.private["time_ns"],
            "시작_기록": state.private["start_record"],
            "이전_기록": records,
            "현재_턴": f"{state.turn + 1}/{state.max_turns}",
            "최고_순위": (None if best is None
                       else {"순위": best, "상위백분위": self._pct(best, n)}),
            "출력_형식": {_OUTPUT_KEY: "<한 글자 이상의 한국어 단어 하나>"},
            "지시": (f'추측할 단어 하나를 위 출력_형식과 똑같은 키를 가진 JSON 오브젝트 '
                   f'하나로만 출력하세요. 예: {{"{_OUTPUT_KEY}": "바다"}}'),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def parse(self, text: str) -> Action:
        raw = text or ""
        # 코드펜스·전후 잡담 허용하고 최상위 JSON 오브젝트만 추출. 정확히 1개 강제.
        dicts = []
        for chunk in _extract_json_objects(raw):
            try:
                value = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                dicts.append(value)
        if not dicts:
            return Action("guess", "", raw, False,
                          f'JSON 오브젝트를 찾지 못했습니다 — '
                          f'{{"{_OUTPUT_KEY}": "<단어>"}} 형식 하나만 출력하세요')
        if len(dicts) > 1:
            return Action("guess", "", raw, False,
                          f'JSON 오브젝트가 여러 개입니다 — '
                          f'{{"{_OUTPUT_KEY}": "<단어>"}} 하나만 출력하세요')
        obj = dicts[0]
        if _OUTPUT_KEY not in obj:
            return Action("guess", "", raw, False,
                          f'"{_OUTPUT_KEY}" 키가 필요합니다 — '
                          f'{{"{_OUTPUT_KEY}": "<단어>"}} 형식으로 출력하세요')
        word = obj[_OUTPUT_KEY]
        if not isinstance(word, str):
            return Action("guess", "", raw, False,
                          f'"{_OUTPUT_KEY}" 값은 문자열이어야 합니다')
        word = word.strip()
        if not _WORD.match(word):
            return Action("guess", "", raw, False,
                          f'"{_OUTPUT_KEY}" 값은 한 글자 이상의 한국어 단어 하나여야 합니다')
        return Action("guess", word, raw)

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
            # 시작_기록(reset에서 발급, 턴으로 안 변함) → episode_end 기록 + verify 재도출 대조.
            "시작_기록": state.private.get("start_record"),
            "stop_reason": state.stop_reason,
        }

    def progress(self, state: GameState) -> dict:
        """누적 진행 지표 — 유효 턴 rank의 최소값(없으면 None). 빈 상태 안전."""
        ranks = [e["rank"] for e in state.history if e.get("valid") and "rank" in e]
        return {"best_rank": min(ranks) if ranks else None}

    def summary_stats(self, episode_ends: list[dict]) -> dict:
        """semantle 전용 집계(고착 지표 중앙값). 기존 summary.json 키를 그대로 보존."""
        plateaus = [e["max_plateau"] for e in episode_ends
                    if e.get("max_plateau") is not None]
        fix_sims = [e["fixation_sim"] for e in episode_ends
                    if e.get("fixation_sim") is not None]
        return {
            "median_max_plateau": statistics.median(plateaus) if plateaus else None,
            "median_fixation_sim": (round(statistics.median(fix_sims), 8)
                                    if fix_sims else None),
        }
