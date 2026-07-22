#!/usr/bin/env python3
"""ko-semantle 어휘·정답 풀 1회성 생성 스크립트 (semantle 2.0.0).

목적: 하드코딩 175어휘/50정답(전부 최빈 명사 → 너무 쉬움)을 대체한다.
- 참조 어휘(REFERENCE): wordfreq 한국어 빈도 상위 토큰을 kiwipiepy로 형태소 분석해
  **일반명사(NNG) 단일 토큰**만 남기고(한글 1~4자, 고유명사/조사/어미/복합 배제) 상위 5,000개.
- 정답 풀(TARGET): 필터된 명사 빈도 순위 300~2,500 구간에서 결정론 400개 선정(최빈어 제외
  = 난이도 상승). TARGET ⊆ REFERENCE 불변식(정답은 참조 어휘에서 뽑으므로 자동 성립).

의존성은 **생성 시에만** 필요(런타임 아님): `python3 -m pip install wordfreq kiwipiepy`.
런타임은 bench/games/data/*.txt만 읽는다. 재현성을 위해 스크립트·데이터 파일 모두 커밋.

생성물 sha256(직접 실행 시 갱신 — 아래 주석은 최근 실행 값):
  ko_vocab.txt   : 0625b01935376c0e22bb4c4fe523793ed102eda3641ee6b8c38b07a91a2648a1
  ko_targets.txt : 6412451369682deba70ead36d6de6ef77fcb9cf599589cbe6dfd52fd9fbfad72
사용법: python3 scripts/gen_ko_vocab.py  (bench/games/data/에 덮어씀 + sha256 출력)
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

from kiwipiepy import Kiwi
from wordfreq import top_n_list

# 길이 2~4자로 제한(브리프 "1~4자"에서 하향): 단일 한글자 최빈어는 조사 동형어(道/銀/義/課
# 등 하네스가 NNG로 태깅)와 외래어 파편(겟/랩/폼/플)에 오염돼 품질이 무너진다(실측). 2자
# 이상만 남기면 일반명사 순도가 크게 오른다.
_HANGUL_2_4 = re.compile(r"^[가-힣]{2,4}$")
# 명백히 민감·비명사인 소수 항목만 수기 배제(고유명사/외설/식별어). NNG 필터가 대부분
# 거르지만 잔여를 정리한다(재현성 위해 스크립트에 고정).
_STOPWORDS = {
    # 민감·외설
    "게이", "레즈", "년놈", "새끼", "존나", "씨발", "지랄", "병신", "미친",
    "자지", "보지", "섹스", "야동", "포르노", "콜걸",
    # NNG 오태깅 잔여(조사/파편/고유명사류) — 육안 확인분
    "로써", "카이", "대요", "스리",
}
VOCAB_SIZE = 5000
TARGET_RANK_LO, TARGET_RANK_HI = 300, 2500   # 필터된 명사 빈도 순위 구간(최빈어 제외)
TARGET_COUNT = 400
TARGET_SEED = 20260716                        # 정답 표본 결정론 seed
# 정답 전용 stoplist — 400개 전수 검수 제외분. 참조 어휘(ko_vocab.txt)에는 남긴다
# (순위 잣대로는 무해하고 vocab_digest·임베딩 캐시 불변 유지가 중요). 정답 풀에서만 뺀다.
_TARGET_STOPWORDS = {
    # 오태깅·어근 파편
    "건지", "대하", "전하", "맞이", "수록", "불구", "강하", "얼마",
    # 형용사 어근(하다-파생)
    "독특", "유명", "유용", "우수", "특이", "피곤", "철저",
    # 외래어 파편·접두어성
    "러브", "데이", "미니", "멀티", "소프트", "플러스", "헤어", "패스", "타임",
    "비트", "모드", "레전드", "해피",
    # 복합어 파편 추정·민감
    "성소", "본격", "섹시",
    # 2차 검수: 비표준 표기(메세지 — 바른 표기 '메시지' 추측이 문자열 불일치로 오답 처리되는
    # 함정)·외래어 파편(월드컵/메이저리그 파편)
    "메세지", "월드", "메이저",
    # 3차 검수: 동사 활용형 오태깅(정한), 지시적 시간어(정답 부적절), 외래어 파편, 형용사 어근
    "정한", "이날", "한때", "당장", "더블", "동일",
    # 4차 검수: 외래어 파편(골드), 접두어성 파편(통행불가/정규직류 — 단독 정답으로 부적절)
    "골드", "불가", "정규",
    # 5차 검수: 동사 어근 오태깅(거두다 — 巨頭 명사 빈도 아님)
    "거두",
}
OUT_DIR = Path(__file__).resolve().parent.parent / "bench" / "games" / "data"


def _filtered_common_nouns(limit: int) -> list[str]:
    """wordfreq 빈도 순 상위 토큰 → 일반명사(NNG) 단일 토큰만 빈도 순으로 반환.

    kiwipiepy 분석 결과가 정확히 한 개 토큰이고 그 태그가 NNG(일반명사)이며 형태가 원단어와
    같은 경우만 채택 → 고유명사(NNP)·조사·어미·복합/굴절어를 배제한다(한글 1~4자).
    """
    kiwi = Kiwi()
    # 넉넉히 큰 후보 풀에서 필터(명사 비율을 감안해 목표의 수 배를 훑는다).
    candidates = top_n_list("ko", limit * 12)
    nouns: list[str] = []
    seen: set[str] = set()
    for word in candidates:
        if word in seen or word in _STOPWORDS or not _HANGUL_2_4.match(word):
            continue
        toks = kiwi.tokenize(word)
        if len(toks) == 1 and toks[0].tag == "NNG" and toks[0].form == word:
            nouns.append(word)
            seen.add(word)
            if len(nouns) >= limit:
                break
    return nouns


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nouns = _filtered_common_nouns(VOCAB_SIZE)
    assert len(nouns) == VOCAB_SIZE, f"명사 부족: {len(nouns)}"

    # 정답 풀: 빈도 순위 300~2,500 구간에서 결정론 표본 400개(빈도 스프레드 확보).
    # 정답 400 선정 — stoplist 안정 불변식: "stoplist에 단어를 추가해도 제외된 단어만 바뀌고
    # 나머지 정답은 절대 안 바뀐다." 이를 위해 stoplist와 무관한 '고정 후보열'(seed 고정 비복원
    # 순차 추출 = 밴드의 결정론 순열)을 한 번 만들고, 그 순서대로 stoplist를 건너뛰며 앞 400개를
    # 취한다. stoplist에 단어를 추가하면 그 단어 자리만 꼬리 후보가 채우고(RNG 스트림이 stoplist
    # 변화에 밀리지 않음), 다른 정답은 순서·포함이 그대로 유지된다.
    band = nouns[TARGET_RANK_LO:TARGET_RANK_HI]
    candidates = random.Random(TARGET_SEED).sample(band, len(band))   # 고정 후보열(prefix-stable)
    picked: list[str] = []
    for w in candidates:
        if w in _TARGET_STOPWORDS:
            continue
        picked.append(w)
        if len(picked) == TARGET_COUNT:
            break
    targets = sorted(picked)

    vocab_path = OUT_DIR / "ko_vocab.txt"
    targets_path = OUT_DIR / "ko_targets.txt"
    vocab_path.write_text("\n".join(nouns) + "\n", encoding="utf-8")
    targets_path.write_text("\n".join(targets) + "\n", encoding="utf-8")

    # 불변식 자기검증
    assert set(targets) <= set(nouns), "TARGET ⊄ REFERENCE"
    assert len(targets) == TARGET_COUNT, f"정답 수 불일치: {len(targets)}"
    assert all(_HANGUL_2_4.match(w) for w in nouns), "비한글/길이 위반"
    assert not (set(nouns) & _STOPWORDS), "불용어 누출"
    assert not (set(targets) & _TARGET_STOPWORDS), "정답 stoplist 누출"

    print(f"ko_vocab.txt   : {len(nouns)} words  sha256={_sha256(vocab_path)}")
    print(f"ko_targets.txt : {len(targets)} words sha256={_sha256(targets_path)}")
    print("sample targets:", " ".join(targets[:30]))


if __name__ == "__main__":
    main()
