# 04 — 창의력 기계 채점 딥리서치 (2026-07-10)

딥리서치 하네스 1회(5각도 fan-out → 20소스 → 100주장 추출 → 상위 25 적대검증 3표: **24 확정 / 1 기각**).
목적: 창의력 전용 벤치를 위한 기계적(judge-free) 채점법 전수 조사 — `03-research.md` 이후,
창의축 재설계 스펙(2026-07-02)의 열린 질문 4개 타깃.

---

## 헤드라인 5개

1. **~0.2–0.3 임베딩 천장은 깨졌다.** 열쇠 두 개:
   - **문맥 임베딩 전쌍 거리(DSI)** — 단문 서사 r=.77(잠재변수 r=.85). 정적 임베딩(GloVe/LSA) 같은 방법은 r=.54 → 문맥 임베딩이 +.15~.27 증분.
   - **풀 상대 역균질화** — 같은 프롬프트의 다른 응답들과의 평균 임베딩 거리(gte-large)가 전문가 평정 최강 양(+) 예측자 β=.56.
2. **LLM-judge는 창의성에서 방향 자체가 틀렸다(실증).** Claude/Gemini/GPT-4 judge는 의미 다양성 β=−.58, novelty β=−.21로 **음(-)** 가중, 어휘 복잡성에 +.42 — 전문가 인간(+.56)과 정반대. → judge-free 원칙이 "편향 회피"를 넘어 **구성타당도 우위**로 정당화됨.
3. **novelty 단독 채점은 게임된다(실증).** WordNet 명사 무작위 추출이 GPT-4~4.5/Claude 3~3.7/Gemini/Llama 전 계열을 DAT에서 이김(온도 무관, GloVe·SBERT 양쪽 재현). → 적절성 게이트 없는 거리 지표는 랭킹 무효.
4. **novelty×적절성 결합: 곱의 실증 근거는 없다.** 실증 구현된 유일 형태는 **통계적 게이트**(CDAT: 큐-코사인 적절성 분포가 무작위 베이스라인을 Welch t + FDR α=0.001로 초과할 때만 novelty 보고). 곱 vs 기하평균 vs 게이트 직접 비교 실험은 부재.
5. **웹 코퍼스 대조(Creativity Index)는 창작 도메인 한정.** 인간 작가가 LLM보다 CI 66.2%↑, RLHF가 CI −30.1%(집단판별) — 그러나 비관행 문제해결(0.76 vs 0.75)·논문 아이디어(0.71 vs 0.71) 판별 실패. n-gram 신규성 상위 사분위의 **~91%가 전문가 기준 비창의**(정밀도 상한).

---

## 방법별 카드

| 방법 | 원리 | 타당도 수치 | 로컬 재현 | 게임 취약점 |
|---|---|---|---|---|
| **DSI** (Johnson 2022, BRM) | 텍스트 내 모든 단어의 BERT-large **6-7층 문맥 임베딩** 전쌍 코사인 거리 평균 | 단문(~60단어) r=.77 [.70,.82], 잠재 r=.85. **길이 붕괴**: ~442단어에서 r=.35 | ◎ 로컬 BERT, 코드 공개(osf.io/ath2s), 결정론 | novelty-only 계열 → 적절성 게이트 필요. 길이 민감 |
| **역균질화** (Ismayilzada, ICCC 2025) | 동일 프롬프트 풀 내 다른 응답들과의 평균 임베딩 거리(1−cos, gte-large) | 전문가 평정 β=.56 (t=14.65) — surprise·어휘다양성(각 β=.09) 압도 | ◎ 임베딩만 (현행 LOF와 동계열) | 참조 풀 의존, 헛소리로 거리 벌리기 → 게이트 필수 |
| **Creativity Index / DJ-Search** (Lu, ICLR 2025 Oral) | 웹코퍼스(Infini-gram) 스니펫 재구성 난이도. near-verbatim 모드 = BM25 후보 + **WMD** | 집단판별만(66.2%/−30.1%), 인간상관 계수 없음. 표현 수준 정밀도 낮음(91% 비창의) | △ Infini-gram API 의존, WMD O(\|d\|²\|w\|) 고비용 | n-gram 범위(5-7→5-11) 설정에 격차 소멸, 코퍼스 컷오프(2020-23) 이후 데이터 누수, 창작 외 도메인 무력 |
| **CDAT 게이트** (EACL 2026 Findings) | 적절성(큐-코사인) 분포가 무작위 베이스라인 초과 시에만 novelty 보고(Welch t + FDR) | 게임 방어 설계 실증(무작위 샘플링 공격 차단 목적) | ◎ 임베딩+stdlib | 게이트가 실제 LLM에 발동된 적 없음(무작위 베이스라인만 배제) |
| **Forward flow** (Gray 2019) | 각 사고의 모든 선행 사고와의 평균 의미거리 | 산출물 채점 r=.19(N=1,397) — 천장 하단. 사람-수준 예측은 β=.48/.42(SEM) | ◎ 쉬움 | **원저자 인정**: 무작위 단어로 점수 상승 → 적대 방어 없음 |
| n-gram novelty 단독 | 참조 LM/코퍼스 perplexity·신규성 | 실재하나 약함(OR≈1.96/SD). LLM에선 novelty↑=실용성↓(OLMo-2 β=−0.48), 인간엔 없는 퇴행 패턴 | ◎ | 신규성 최대화가 곧 퇴행 — 단독 사용 부적합 |

---

## 우리 설계에 대한 함의

- **현행 LOF 풀 희소성 = 역균질화의 국소 버전.** 가장 강한 실증 신호(β=.56)와 동계열 — 재설계 방향이 옳았다는 외부 실증. 유지.
- **DSI를 아이템 내부 지표로 추가할 가치.** 우리 probe 항목은 단문(아이디어 한 줄~비유 한 단락) = DSI의 최적 길이대. LOF(풀 상대, 항목 간)와 DSI(문맥 통합, 항목 내)는 상보적. BERT류 로컬 모델 필요(ollama 임베딩 외 추가 의존성).
- **온토픽·검증 게이트 필수 재확인.** 무작위-샘플링 공격 실증이 우리 곱셈 게이트 설계의 존재 이유를 입증. 단 "곱이 최적"이라는 근거는 없음 → CDAT식 통계 게이트(분포 비교) 전환은 v2 검토 사항.
- **CI/n-gram 대조는 tech(아이디어 발산) 서브에 부적합**(도메인 일반화 실패 실증). copy/humor 같은 언어 표면 창의엔 보조 신호 가능하나 정밀도 상한(91%)과 비용 감안 시 우선순위 낮음.
- **RLHF 정렬도가 교란변수**(CI −30.1%): 정렬 강도가 다른 모델 간 비교 시 "창의력 차이"가 정렬 차이일 수 있음 — 리포트 해석 주의 문구 추가 가치.
- **패러프레이즈 사이비 독창**: 현존 최선례는 DJ-Search near-verbatim(BM25+WMD)이지만 bag-of-words라 구조 변형에 이론상 취약 + 고비용. 완전 해법 부재(열린 질문 유지). NoveltyBench의 기능적 동등성 분할(경량 로컬 분류기)이 미검증 리드.

## 기각/한정된 주장 (정직 표시)

- ✗ "DSI가 장문 창작물 타당도 증거" — 적대검증 1-2 기각. DSI는 장문에서 붕괴(r=.35)하며, 역균질화 증거도 5문장 스토리 한정. **500단어+ 장문의 r≥.5 기계 채점법은 이번 조사에서 발견되지 않음.**
- ⚠ r=.85는 측정오차 보정 잠재변수 추정치, r=.77은 최적 조건(5문장, 5인 평정 집계) 최고치 — 논문 내 범위 .35~.77.
- ⚠ "문맥 임베딩이 천장 돌파의 열쇠"는 2-1 통과(medium) — 서사 과제 한정, AUT/DAT 직접 검증 없음.
- ⚠ 대부분 발견이 단일 1차 논문 의존(원문 verbatim 대조는 통과). CDAT·Death of the Novel(ty)·Rethinking CI는 2026년 게재 최신 논문 — 후속 반박 가능성.

## 열린 질문 (후속 추적)

1. novelty×적절성의 곱 vs 기하평균 vs 게이트를 인간 평정 기준으로 직접 비교한 연구 — 부재.
2. 500단어+ 장문에서 r≥.5 기계 채점법 — 부재. 청킹 후 DSI/문단 수준 역균질화의 타당도 미검증.
3. 구조적(통사 재배열) 패러프레이즈까지 잡는 기계 지표 — 부재.
4. TTCW·NoveltyBench·CS4·LiveIdeaBench 채점 컴포넌트의 judge-free 분리 전용 가능성 — 확정 클레임 생존 0건(특히 NoveltyBench distinctness 분할의 임베딩 재현 시 인간상관 유지 여부).

## 인용

- DSI 원논문: https://pmc.ncbi.nlm.nih.gov/articles/PMC10615993/ (Johnson et al. 2022, Behavior Research Methods)
- 역균질화·LLM-judge 편향: https://arxiv.org/pdf/2411.02316 (Ismayilzada, Stevenson & van der Plas, ICCC 2025)
- DAT 무작위 베이스라인·CDAT 게이트: https://arxiv.org/pdf/2601.20546 (EACL 2026 Findings) + https://arxiv.org/abs/2310.11158 (Chen & Ding, EMNLP 2023 Findings)
- Creativity Index: https://arxiv.org/abs/2410.04265 (Lu et al., ICLR 2025 Oral) / 코드 https://github.com/GXimingLu/creativity_index
- n-gram 신규성 한계: https://arxiv.org/abs/2509.22641 (Death of the Novel(ty), ICLR 2026)
- CI 도메인 일반화 실패: https://arxiv.org/abs/2508.05470 (Rethinking Creativity Evaluation, EACL 2026)
- Forward flow: Gray et al. 2019 (American Psychologist) + Beaty et al. 2021 (Thinking Skills and Creativity)
