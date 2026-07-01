# 창의·발산 축 재설계 — 독창성 중심(judge-free, 순수 기계 채점)

- 상태: 설계 승인 → 스펙 (2026-07-02)
- 대상 코드: `bench/axes/creativity.py`, `bench/probes.py`, `bench/embed.py`, `bench/report.py`
- 관련 원칙: README 설계 3원칙(judge-free / 오염방어 / 기저분리), `docs/02-methodology.md`

## 1. 문제

현행 창의축(`axes/creativity.py`)은 **순수 내부 발산(MPD, 한 응답 내 항목들끼리의 평균 쌍별 거리)** 중심이다. 결함:

- **내부 발산 ≠ 독창성.** "서로 다른 뻔한 것들"(예: RAG 표준기법 12개 나열)은 내부 발산 만점이지만 전부 클리셰다. 실측에서 4.8/4.6/s5가 **52~57 좁은 밴드에 뭉쳐 변별이 안 됨**(2026-07-02 run).
- 품질/독창 게이트가 얇음(개수 기반 validity·fluency뿐, 반-클리셰 신호 없음).
- toy probe(DAT 무관단어 / AUT 사물용도)가 사용자 실사용(브레인스토밍·카피·유머·비유)과 괴리.

## 2. 목표

**"독창성 = 남들이 몰린 데서 벗어나되(희소) 주제는 지킨(온토픽) 아이디어"** 를 **판단하는 LLM 없이** 재는 축. 등장하는 LLM은 오직 피험자(답 생성)뿐, 채점은 전부 고정 임베딩 + stdlib 산수(결정론적).

부차: 사용자 가설 **"창의력↑ → 말투↑"** 를 창의축 vs 청중축(audience) 상관으로 리포트에 부가 검증(축엔 섞지 않음).

## 3. 설계 결정 (딥리서치 근거)

각 결정은 2026-07-02 딥리서치(주장 22/25 검증통과)에 근거. 기각된 특정 상관 수치는 인용하지 않음.

| 결정 | 근거 |
|---|---|
| 독창 = **중심거리 아님, 국소밀도 희소성(LOF)** | LOF 밀도비율은 클러스터별 밀도차에 강건 — 온토픽 아이디어를 국소 이웃 대비 독창으로 포착. 단일 global centroid 거리보다 우월 (scikit-learn; arXiv 2411.02738) |
| 항목 집계 = **MAX(최고 독창 1개)**, 합/평균 아님 | 유창성/양치기 confound 제거. max scoring이 타당도 최고 (Beaty 2022, CRJ) |
| 온토픽 게이트 = **probe별 상대 정규화 코사인**, 고정 커트라인 금지 | 대조학습 임베딩 코사인은 자연 커트라인 없음 → 절대상수 엉터리. query-dependent 정규화 원칙만 judge-free 채택 (Rossi 2024, CIKM) |
| 반게임 = **gzip압축률 + 긴 n-gram 자기반복 + Self-BLEU + MTLD** | 상호 상관 낮은 비중복 조합, 전부 stdlib. MTLD가 길이에 제일 강건. **모두 길이 편향 → 길이 정규화 필수** (Shaib 2024; McCarthy&Jarvis 2010) |
| 곱셈 게이트(가산 평균 아님) | novelty-only는 randomness와 창의를 혼동 → 헛소리가 거리 만점. 온토픽·검증이 0이면 거리 무의미하게 곱으로 소거 (Gray 2019; Nakajima 2026) |
| 단일모델 폴백 = **CDAT형**(내부발산 × 온토픽) | 풀 없을 때: novelty=내부 쌍별거리, appropriateness=프롬프트 cue 거리 penalize (Nakajima 2026, EACL) |
| 정규화 = **풀 상대 백분위**, 자의적 상수 0 | README 정규화 원칙 부합 |

**정직한 천장(스펙에 명시):** 순수 임베딩 독창성은 인간 평가와 상관 **~0.2–0.3**에 그치고, 임베딩 모델을 키워도 안 오른다(거리 가정 자체의 한계, Organisciak 2023). → 점수는 **정밀 등급이 아닌 거친 순위**로만 해석. 리포트·note에 경고 문구 고정.

## 4. Probe 설계 (`probes.py`)

4개 서브타입, 각 서브 N개. 오염방어: 주제/엔티티를 seed로 매 실행 난수화. 포맷 지시(번호 목록, 한 줄 1항목)로 파싱 안정화. "독창적으로/참신하게" 지시는 **넣는다**(창의는 명시적으로 요구하는 게 과업 자연스러움) — 단 "뻔한 답 피하라"까지만, 채점 기준은 노출 안 함.

| 서브 | 프롬프트 shape | 난수화 | 클리셰(=풀 밀집) 예 |
|---|---|---|---|
| `tech` 기술발산 | "표준 파이프라인 넘어 [도메인]의 참신한 아이디어 12개, textbook 금지" | 도메인 풀(RAG·에이전트메모리·불확실성·캐싱·평가…) | 청킹튜닝·리랭커·하이브리드검색 |
| `copy` 카피/네이밍 | "[제품컨셉]의 이름/태그라인 10개" | 컨셉 풀 | smart/pro/hub/AI-prefix |
| `humor` 유머/풍자 | "[상황]에 대한 위트/풍자 10개" | 상황 풀 | 미지근한 말장난 |
| `metaphor` 비유/설명 | "[개념]을 참신한 비유로 설명" | 개념 풀 | CPU=두뇌, 인터넷=고속도로 |

기존 `divergence_probes`(rag/uncertainty/agent_memory/llm_eval)를 `tech` 서브로 흡수. `metaphor`는 정확성 요구가 커서 온토픽 게이트(개념과의 코사인)로 근사하되 **최약축으로 스펙에 명시**(개선 여지: 개념 정합 별도 지표).

## 5. 채점 파이프라인 (`axes/creativity.py`)

### 5.1 항목 추출
응답 텍스트 → `parse_items`(기존 재사용, 번호/불릿 파싱). 서브 `metaphor`는 응답 전체가 1항목.

### 5.2 풀 인지(POOL-AWARE) — 아키텍처 변경
현행 `score(axis, samples)`는 **모델 1개** 샘플만 받는다. LOF 희소성은 **한 probe의 전 모델 아이디어를 한 공간**에 놓아야 하므로, 창의축은 **풀 인지 경로**가 필요.

- `report.score_run`에 창의축 특례: 해당 축의 `per_model`(전 모델) 전체를 넘기는 pooled 스코어러 호출.
- 신설 인터페이스: `creativity.score_pool(per_model: dict[str, list[Sample]]) -> dict[str, AxisResult]`.
  - (probe, sub)별로 전 모델 아이템을 모아 임베딩 → 풀 구성.
  - 각 아이템의 novelty·gate 계산 → 모델별 결과 반환.
- 레지스트리에 "pooled axis" 플래그 추가(`axes/__init__.py`). 다른 축은 기존 per-model 경로 유지.

### 5.3 아이템 점수
probe별 풀 `P`(전 모델 아이템 임베딩) 위에서, 아이템 `x`:

```
novelty(x)   = LOF_k(x, P)                 # 국소밀도 희소성. k = min(20, |P|-1) 등 데이터적응
ontopic(x)   = zpct( cosine(x, prompt_vec), over P_thisprobe )   # probe 내 상대 백분위(0~1)
validity(x)  = gate( gzip_ratio, long_ngram_rep, self_bleu, mtld )  # 길이보정, 0~1
item(x)      = norm_pct(novelty(x), P) * ontopic(x) * validity(x)
```

- `LOF_k`: 코사인거리 기반 kNN 국소밀도 비율(sklearn 없이 `embed` 위에 구현). 고차원 거리집중 완화 위해 k 적응·거리 표준화.
- `zpct`: 프롬프트 코사인을 **그 probe의 답들 분포**로 백분위화(절대 커트라인 X). 낮은 쪽(딴소리) → ontopic↓.
- `validity` 게이트: 각 지표를 길이보정 후 0~1로, 반복/degeneration 심하면 0쪽. 곱으로 결합.

### 5.4 모델·서브·축 집계
```
sub_score(model, sub) = MAX_x∈model  item(x)          # 최고 독창 1개 (양치기 방어)
model_score(model)    = mean_sub sub_score            # 서브 평균
axis 0–100            = 풀 내 백분위 매핑 (상대)
```

### 5.5 단일모델 폴백 (풀 없음/모델 1개)
LOF 불가 → CDAT형: `novelty = 내부 MPD(모델 자기 아이템들)`, `× ontopic(프롬프트 cue 거리) × validity`. note에 "폴백(상대비교 아님)" 표기.

### 5.6 임베딩 불가
`embed.available()==False` → `AxisResult(score=0, n=0, note="ollama 임베딩 서버 불가")` (기존 동작 유지).

## 6. 지원 유틸 (`embed.py`)
- `knn_cosine_distances(vecs, k)` / `lof(vecs, k)` 추가.
- 반게임 지표는 stdlib 전용 모듈(예: `axes/_textmetrics.py`)에: `gzip_ratio`, `long_ngram_repetition`, `self_bleu`, `mtld`(기존 creativity.mtld 이전), `distinct_n_adjusted`. **전부 길이 정규화 포함.**

## 7. 말투 가설 검증 (`report.py`)
- 창의 `model_score` vs 청중(audience) `score`의 모델별 산점/상관을 리포트에 부가 카드로 표시.
- 축 점수엔 영향 없음(독립 유지). 상관은 **관측**으로만 서술("가설 성립 경향/미성립"), 인과 주장 금지.

## 8. 반게임 분석 (무엇이 깨나)
- **랜덤 헛소리**: 거리 만점이나 ontopic↓ → 곱으로 소거. ✅
- **반복/재탕으로 발산 위장**: gzip·self-BLEU·긴n-gram이 validity↓. ✅
- **양치기(많이 뿌리기)**: MAX 집계라 개수 무의미. ✅
- **패러프레이즈/엔티티만 바꾼 사이비 독창**: 부분 취약 — 임베딩 거리는 표면 다양성에 속을 수 있음. 완화: specificity/엔티티밀도 floor는 **후속 개선**으로 열어둠(openQuestion).
- **고차원 거리집중(768차원)**: LOF k 선택 민감 → k 적응 + 거리 표준화로 완화, 완전 해결은 아님(명시).

## 9. 범위 밖 (YAGNI)
- LLM 심판(제3모델 pairwise) — 명시적으로 제외(judge-free 원칙).
- 런타임 클리셰 생성(haiku obvious-set) — 풀이 클리셰를 자기정의하므로 불필요.
- specificity/엔티티밀도 floor, metaphor 개념정합 지표 — v2 후보.
- 인간 코퍼스 백분위 — 다모델·인간 데이터 확장 시.

## 10. 열린 질문 (연구 미해결, 추적)
1. 장문 답에 대한 LOF/CDAT의 인간 상관 타당도(원 검증은 단어/짧은 과업).
2. 온토픽 임계의 완전 label-free 설정(백분위 knee vs z-score).
3. far×ontopic×validity 결합 최적형(곱 vs 가중 기하평균).
4. 패러프레이즈 사이비 독창을 잡는 순수 기계 의미-신규성 지표.

## 11. 테스트 (구현 시 TDD)
- 합성 케이스: (a) 뻔한-발산(표준기법 나열) → 저독창 확인, (b) 랜덤 헛소리 → ontopic 게이트 0 확인, (c) 반복 padding → validity 0 확인, (d) 진짜 외딴 온토픽 아이디어 → 고점 확인.
- 길이 정규화: 같은 내용 길이만 2배 → 점수 불변(±ε).
- 풀/폴백 경로 동등성 스모크.
- 결정론: 같은 입력·seed → 동일 점수.
