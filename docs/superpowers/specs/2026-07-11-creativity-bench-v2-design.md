# 창의력 단독 벤치 v2 — 장문 궤적 + DSI + 통계 게이트 (judge-free)

- 상태: 설계 승인 → 스펙 (2026-07-11)
- 대상 코드: `bench/probes.py`, `bench/axes/creativity.py`, `bench/axes/creativity_long.py`(신규), `bench/dsi.py`(신규), `bench/axes/__init__.py`, `bench/report.py`, `bench/cli.py`, `gen_blind.py`
- 근거: `docs/04-creativity-research.md`(딥리서치 2026-07-10, 24 확정/1 기각), v1 스펙 `2026-07-02-creativity-axis-redesign-design.md`
- 원칙: README 설계 3원칙(judge-free / 오염방어 / 기저분리), `docs/02-methodology.md`

## 1. 배경/문제

v1(LOF 풀 희소성×온토픽×검증×접지)은 **단문 아이디어 발산만** 측정한다. 딥리서치 결론:

- 임베딩 천장(~0.2–0.3)은 **문맥 임베딩 DSI**(단문 r=.77)와 **풀 상대 역균질화**(전문가 평정 β=.56)로 돌파됨. 후자는 v1 LOF와 동계열 — 방향의 외부 실증.
- **LLM-judge는 창의성에서 역방향 편향**(의미 다양성 β=−.58, 어휘 화려함 +.42 — 전문가와 정반대) → judge-free 유지 확정.
- **novelty 단독은 무작위 샘플링에 패배**(WordNet 무작위가 전 SOTA LLM을 이김) → 게이트 필수.
- **500단어+ 장문의 검증된 기계 채점법은 부재** → 본 스펙의 장문 궤적은 **미검증 신규 시도**이며, 블라인드 검증 카드(§7)가 그 시험대.
- 어휘/토큰 분포 지표는 전문가 신호가 아님(AI심판·비전문가만 속음) → **실격 게이트 전용**, 점수 산입 금지.

## 2. 확정 결정 (2026-07-11 사용자 인터뷰)

| 항목 | 결정 |
|---|---|
| 위치 | 기존 llm-bench 내, creativity 전용 실행 모드 |
| 모델 풀 | 이원화 — 개발/스모크: 클로드 3종(4.8/4.6/sonnet-5, 구독), 정식: 6–9종 멀티벤더(codex·gemini 포함) |
| 소설 길이 | 5,000자 기본(K=15 청크), `--long` 10,000자(K=30) |
| 언어 | 단문 4서브 영어(현행 유지), 장문 소설 **한국어** |
| DSI | 원논문 그대로(PyTorch+transformers **선택** 의존성), 보조점수로 시작 |
| 구조 | 형제 축 분리: `creativity`(단문) + `creativity_long`(장문), 리포트 단일 카드 묶음 |

## 3. 아키텍처

```
bench/
├── probes.py               # [수정] novel 서브 추가(한국어 주제 조합 난수화)
├── axes/
│   ├── __init__.py         # [수정] creativity_long 등록(pooled 플래그)
│   ├── creativity.py       # [수정] DSI 보조점수 연결 + 통계 게이트
│   └── creativity_long.py  # [신규] 장문 궤적 채점(score_pool)
├── dsi.py                  # [신규] DSI 계산(transformers lazy import)
├── report.py               # [수정] 창의력 묶음 카드 + 궤적 SVG + 검증 카드
└── cli.py                  # [수정] --long 플래그
```

- 흐름: `run`(단문 영어 + 소설 한국어 생성) → `score`(단문: v1 파이프라인+DSI+통계게이트 / 장문: 궤적) → `report`(창의력 카드 + 검증 카드).
- 의존성: ollama 임베딩 **필수**(현행), torch+transformers **선택**(없으면 DSI만 생략+note, 나머지 전부 동작).
- 다른 축(density, sycophancy 등)은 불변.

## 4. Probe (`probes.py`)

### novel 서브 (신규)
- 주제 = seed 조합 난수화: `BACKDROPS × CHARACTERS × MOTIFS` 풀(각 ≥8개)에서 뽑아 "폐업 직전 목욕탕을 배경으로, 보험조사원이 등장하고, '열쇠 하나가 사라진다'는 제약을 포함" 형태의 주제 문자열 생성. **전 모델 동일 주제**(풀 상대 채점 성립) + 조합 난수화(암기 오염 차단).
- 지시문(한국어): "다음 주제로 5,000자 이상 단편소설을 써라. 본문만 출력(제목·머리말·메타 코멘트 금지). 뻔한 전개를 피하고 독창적으로." — 창의 요구는 명시, 채점 기준은 비노출(v1 원칙). `--long` 시 10,000자.
- 기본 런: 주제 2개 × 모델당 1생성. meta: `{subtype:"novel", lang:"ko", target_chars, k_chunks}`.
- 단문 4서브(tech/copy/humor/metaphor): 변경 없음(영어, fmt=report 계약 유지).

## 5. 장문 채점 (`axes/creativity_long.py`)

### 5.1 청킹
- 문장 경계(종결부호 기준) 존중, 누적 길이 기준 **균등 K분할**. K=15(5천자)/30(만자). K는 가중치가 아니라 측정 해상도 파라미터 — 전 소설 동일 조각 수로 조각-수 교란 제거, 결정론.
- 조각당 평균 문장 수 < 1이면 `AxisResult(score=0, note="채점 불가(과소 길이)")`.

### 5.2 지표 (임베딩=ollama, 정규화=풀 상대 백분위, 자의 상수 0)
- **progress**: `mean_{i=2..K} min_{j<i} cos_dist(c_i, c_j)` — 이야기가 계속 새 의미 지역으로 이동하는가(맴돌면 하락). 청크 임베딩 전쌍이 아닌 **nearest-previous** 거리. `cos_dist = 1 − cosine`(`embed.py` 관례).
- **rarity**: 같은 주제의 전 모델 청크를 한 공간에 풀링 → LOF(k 적응, v1 `embed.lof` 재사용) → 모델별 **MAX** 청크(양치기 방어, v1과 동일 집계 원리).
- **ontopic**: 주제 프롬프트 벡터 vs 소설 청크 평균 벡터 코사인 → 풀 내 `ontopic_gate`(v1 재사용).
- **validity**(실격 전용): `_textmetrics`를 **어절(공백 토큰) 기반**으로 적용 — gzip_ratio, 긴 n-gram 반복, self-BLEU(청크 간), MTLD. 길이 보정 유지. 형태소 분석기 의존성 없음(한계 명시, §12).

### 5.3 결합/집계
```
novel_score(model, topic) = (pct(progress) + pct(rarity_max)) / 2 × ontopic × validity
model_score = mean over topics → ×100
```
- novelty 계열 2개(progress·rarity)는 산술평균 — 곱/기하평균 대비 실증 부재(열린 질문 §12), 블라인드 상관으로 후속 결정.
- 폴백(모델 1개/풀 없음): rarity 제외, `progress × ontopic × validity` + note "폴백(상대비교 아님)".
- note 고정 문구: "장문 궤적 채점은 인간상관 미검증(연구 공백 영역) — 검증 카드 참조."

### 5.4 지구력 카드 (`--long` 런 전용)
- 전반 K/2 vs 후반 K/2의 progress 평균 비교 → 하락률을 리포트에 표기(점수 비합산).

## 6. 단문 업그레이드 (`axes/creativity.py` + `dsi.py`)

### 6.1 DSI 보조점수
- `dsi.py`: bert-large-uncased **6·7층** 토큰 문맥 임베딩(특수토큰·구두점 제외) 전쌍 cosine distance 평균 — 원논문(Johnson 2022) 그대로. CPU, `eval()` 고정 결정론. lazy import — 미설치 시 note "DSI 생략(transformers 미설치)".
- 적용: `metaphor`=항목 전체(검증 조건 최근접), 기타 서브=항목별 계산 후 평균(**탐색적** — 원 검증 범위보다 짧음, note 표기).
- 출력: `subscores["dsi_<sub>"]`(풀 상대 백분위 ×100). **본점수 비합산** — 블라인드 상관 확인 후 합류 결정.

### 6.2 통계 게이트 (CDAT식, 모델×probe 수준 실격 장치)
- 무작위 베이스라인: 같은 run의 **다른 probe** 아이템들을 현 probe 프롬프트에 코사인 → 주제 무관 텍스트 분포(n≥20 요건, 미달 시 게이트 생략+note).
- 검정: 모델의 해당 probe 아이템 코사인 분포 vs 베이스라인 — Welch t(stdlib 구현, 정규 근사 p 명시), α=0.001, **Bonferroni**(모델×probe 셀 수). 유의 초과 실패 시 그 셀 novelty=0 + detail 표기.
- 기존 아이템 수준 시그모이드 게이트는 유지 — **이중 게이트**.
- 적용 범위: **단문 4서브 전용.** 장문(novel)은 베이스라인 재료(다른 probe 아이템)가 영어라 한국어 소설과의 언어 혼합 분포가 성립하지 않음 — 장문은 온토픽+validity 게이트로 충분(§5).

## 7. 리포트 (`report.py`)

- **창의력 카드**: creativity(단문 서브 막대 + dsi_* 보조) + creativity_long(장문 점수) 묶음. **궤적 라인차트 SVG**: x=청크 순번, y=progress 거리, 모델별 폴리라인 — "어디서 맴돌기 시작했나" 가시화.
- **검증 카드**: `results/raw/blind_pool_creativity/blindrank_result.json` 존재 시 기계점수 vs 본인 블라인드 순위의 **Spearman ρ + n** 그대로 표기(자의적 합격선 없음). 부재 시 "미검증".
- 고정 주의 문구: ① 장문 궤적 미검증 ② RLHF 정렬이 창의 점수를 낮추는 교란(CI −30.1% 실증) ③ v1 천장 문구 유지.
- `gen_blind.py`에 novel 산출물 포함(본인 평정 수집 경로).

## 8. CLI

```bash
python3 -m bench run --axes creativity,creativity_long [--long] [--models …]
python3 -m bench score / report   # pooled 특례 경로 재사용
```

## 9. 게임/오염 방어

| 공격 | 방어 |
|---|---|
| 주제 암기(오염) | 조합 난수화 + 채점 기준 비노출 |
| 무작위/헛소리 텍스트 | 온토픽 게이트 + **통계 게이트** + 접지 게이트(단문) |
| 반복 패딩·재탕 | validity 실격(gzip·n-gram·self-BLEU·MTLD) |
| 양치기(다작) | MAX 집계(rarity), K 고정(progress) |
| 장면마다 랜덤 소재 투입(progress 게임) | 온토픽·validity가 1차 방어 — **완전 해결 아님**, 열린 질문 §12 |
| 채점기 익스플로잇 | 피험 모델은 채점 경로에 등장하지 않음(임베딩·BERT는 고정 로컬 모델) |

정직 표기: 통계 게이트는 원논문에서도 실제 LLM에 발동된 적 없음(무작위 베이스라인 배제용 보험 장치).

## 10. 테스트 (TDD)

- 합성: (a) 같은 장면 맴도는 소설 → progress 하위 (b) 주제 무시 랜덤 한국어 → ontopic 소거 (c) 어절 반복 패딩 → validity 실격 (d) 진짜 전개 있는 소설 → 상위 확인.
- 불변식: 같은 내용 길이 2배 → 점수 불변(±ε) / 같은 입력·seed → 동일 점수(결정론) / 청킹: 문장 수 < K 경계, K조각 수 보장.
- 게이트: 무작위 베이스라인 주입 모델 → 통계 게이트 발동 / n<20 → 게이트 생략+note.
- DSI: transformers 미설치 → 우아한 생략 / 설치 시 고정 입력 회귀값 스냅샷.
- 경로: 풀/폴백 동등성 스모크, creativity_long 레지스트리 등록.

## 11. 범위 밖 (YAGNI)

- 한국어 형태소 분석기 의존성 · 임베딩 2D 투영 시각화 · DSI 본점수 합류(블라인드 검증 후 별도 결정) · 인간 코퍼스 백분위 · LLM 심판(영구 제외).

## 12. 열린 질문 (추적)

1. progress+rarity 결합형(평균 vs 곱 vs 기하평균) — 실증 부재, 블라인드 상관으로 결정.
2. 장문 궤적의 인간상관 — 본 벤치의 블라인드 카드가 사실상 최초 검증.
3. 한 줄 아이템 DSI 타당도(원 검증 범위 밖).
4. 어절 기반 validity 지표의 한국어 강건성.
5. "장면마다 랜덤 소재" progress 게임의 완전 방어.

## 13. 수용 기준

1. §10 전 테스트 통과.
2. 클로드 3종 개발 런 end-to-end(run→score→report) + 창의력 카드·궤적 SVG·검증 카드(미검증 표기) 생성.
3. 블라인드 평정 수집 후 검증 카드에 ρ·n 표기 동작(별도 세션에서 평정 수행).
