# 03 — 리서치 결론 + 인용

딥리서치 하네스 2회(소스 fan-out → 적대적 검증 2/3 반박 시 폐기) + Gemini 3.1 Pro 자문 2회.

---

## 딥리서치 1차 — 창의성 정량화 / 오라클 없는 채점

### 핵심 결론
- **judge-free ≠ oracle-free** (헷갈리면 설계 흔들림)
  - judge-free: LLM 심판 안 씀(정답은 있을 수 있음). 예: LiveBench
  - oracle-free: 정답 자체가 없음. 예: 창의 발산 축
- **품질 × 다양성을 반드시 같이 재라.** 한 축만 재면 게임됨:
  - BLEU(품질) → 반복으로 게임
  - self-BLEU/distinct-n(다양성) → 횡설수설이 만점
  - 정도(正道): **분포 거리**(MAUVE류) 또는 품질+다양성 결합
- **DAT(Divergent Association Task)**: 7개 단어의 21쌍 GloVe **의미 거리** 평균 ×100. 사람 r≈0.50. 단 희귀어/운율어로 게임 가능, 좁은 구성개념. 최상위 LLM도 평균적 인간 수준.
  - ⚠️ 통념 함정(검증 폐기): "cosine **유사도**" (X, 거리임), "DSI = BERT cosine 유사도" 정의(X, 폐기) → 쓸 거면 원논문 확인
- **Semantic Entropy**: 레퍼런스 없이 의미 클러스터 엔트로피, MacGyver κ=0.56 > cosine 0.49 (신뢰도 중간)

### 인용
- DAT: https://www.pnas.org/doi/10.1073/pnas.2022340118 , https://www.nature.com/articles/s41598-025-25157-3
- self-BLEU: https://arxiv.org/abs/1904.03971
- 분포 거리/품질-다양성: https://arxiv.org/abs/2007.01488
- judge-free n-gram(0.9896): https://arxiv.org/abs/2502.09316
- LiveBench(judge-free): https://livebench.ai/livebench.pdf
- LLM judge 편향(오류 46%): https://arxiv.org/abs/2410.21819

---

## 딥리서치 2차 — 능력차원 / 채점메커니즘 / 오염방어 / 통계

### (1) 능력 차원 마스터 분류
- 지식·능력 / 정렬 / 안전 3대 그룹: https://arxiv.org/pdf/2310.19736
- 283개 벤치 3계층(일반/도메인/타깃): https://arxiv.org/pdf/2508.15361

### (2) 채점 메커니즘 4패밀리
규칙기반(exact/BLEU/ROUGE) · 신경/임베딩(BERTScore/BLEURT/BARTScore) · 인간 · LLM 판정.
- BLEU/ROUGE는 의미 등가 못 잡음 → 임베딩 지표 등장
- FActScore: atomic fact 분해 → 지지율 (분해-검증)
- LLM-judge 프로토콜: pointwise/pairwise/listwise. Chatbot Arena = pairwise→Elo(현 Bradley-Terry MLE)
- 인용: https://arxiv.org/pdf/2310.07521 , https://arxiv.org/abs/2305.14251(FActScore) , https://arxiv.org/abs/2306.05685(MT-Bench) , https://arxiv.org/abs/1904.09675(BERTScore)

### (3) 오염·Goodhart 방어 — 가장 중요
- 💣 **UC Berkeley(2026-04): 검사한 모든 주요 에이전트 벤치가 0태스크 해결로 ~100% 익스플로잇 가능** (구성타당도 실패)
  - SWE-bench Verified: 10줄 conftest.py로 전 인스턴스 "해결"(채점기가 모델 제어 컨테이너 pytest 출력 신뢰)
  - WebArena: 태스크 설정에 정답 동봉, file://로 ~100%
  - WebArena/OSWorld: 모델 제어 문자열에 eval() 호출
  - 인용: https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/ , https://arxiv.org/abs/2605.12673
- **시점분할**(LiveCodeBench): 진행 중 콘테스트에서 수집, 정답 공개 전 평가 — 오염 구조적 차단. https://arxiv.org/abs/2403.07974
- **ABC 체크리스트**: 설계 결함이 성능 추정을 상대적 최대 100% 왜곡. task validity vs outcome validity 분리. https://arxiv.org/abs/2507.02825

### (4) 통계적 엄밀성
- **LongScore**: 장문맥 점수를 단문맥 baseline으로 정규화 → RULER에서 상위 1·2위 역전. 두 모델 비교 시 기저능력 먼저 분리. https://aclanthology.org/2025.findings-acl.903.pdf
- Elo/Bradley-Terry: 적은 N에서 raw 점수보다 안정적 비교.
- 빈칸(미확정, 후속 조사 필요): 적은 N 부트스트랩/검정력 실무 절차, 메타모픽/property-based 구체 규칙, 캘리브레이션 표준 지표(ECE).

---

## Gemini 자문 (3.1 Pro High, `agy --model "Gemini 3.1 Pro (High)"`)

### 1차 — 신박 측정법
- 절차적 제약 기반 서사(씨앗-페이오프 소설), MTLD+클리셰 패널티, 추론 동형성 보존율(언어격차), 추가축: 지적겸손/지시망각/과도한거부.

### 2차 — 우리 설계 비판 + 티어링
- 채택: **gzip 압축률**(재탕 객관 탐지), **종결어미/품사 분포**(정규식 대안), **behavioral diff**(아첨 객관화), **사과세**, **파괴적 리팩토링 삭제력**, **오류회복 MTTR**.
- 반려: 창의발산 Tier2 강등(반려), 청중적응을 라포에 통합(부분 반려 — 객관/주관 분리).

---

## 폐기된 주장 (정직 표시)
- DAT "cosine 유사도" framing (거리가 맞음)
- DSI "BERT cosine 유사도" 정의
- "MCQA accuracy가 지식벤치 지배적 형식"
- OSWorld 인간-모델 격차 수치(72.36% vs 12.24%)
- 2310.19736 출처의 오염방어 framing
