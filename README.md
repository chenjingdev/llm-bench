# llm-bench

공개 LLM 리더보드(SWE-bench, Chatbot Arena 등)를 불신하는 개발자를 위한 **개인용 N-of-1 모델 비교 벤치마크**.
"어느 모델이 천장 능력이 높은가"가 아니라 **"내 실사용에 어느 모델이 맞는가(fit)"**를 다축으로 측정한다.

## 왜 만드나
- 리더보드는 오염(contamination)·Goodhart·하네스 과적합에 취약하고, "나랑 일할 맛"을 안 잰다.
- 2026 UC Berkeley 연구: 주요 에이전트 벤치 전부가 **0태스크 해결로 ~100% 점수** 익스플로잇 가능 → 점수보다 채점기 설계가 먼저.
- 그래서 직접, 작고, 사적이고, 신선한(오염 없는) 벤치를 만든다.

## 1차 비교 대상
- **Claude Opus 4.8 vs 4.6** (가설: 4.8 = 실행/추론, 4.6 = 대화/발상)
- 확장: 향후 모든 신모델을 같은 하네스로

## 핵심 산출물
- **8축 레이더(다각형) 차트** — 두 모델 다각형을 겹쳐 "성격 모양"을 한눈에
- 보조: Tier 2 진단표(캘리브레이션·과도한거부·언어격차 등)

## 설계 3원칙
1. **judge-free 기본** — LLM-as-judge는 편향(자기선호·장황함·위치, 오류율 46%) 때문에 최후수단. 특히 비교 대상 모델로 비교 대상을 채점 금지.
2. **오염 방어** — 차등 퍼징(랜덤 입력)·시점분할·난수 엔티티로 암기 무력화. 채점기는 평가 대상과 격리, 정답을 모델이 읽을 수 있는 곳에 두지 않음.
3. **기저능력 분리** — 축마다 baseline 정규화(LongScore 교훈). 자의적 상수 금지, 상대비교/백분위 사용.

## 현재 상태 (2026-06-18)
- [x] 방법론 딥리서치 2회 + Gemini 자문 2회 완료 → `docs/03-research.md`
- [x] 8축 레이더 설계 + Tier 분류 → `docs/01-design.md`
- [x] 축별 측정 스펙 초안 + "자폐 묶음" 3증상 측정안 → `docs/02-methodology.md`
- [x] **러너 하네스** — `claude -p` 구독 격리 호출(effort 고정·설정 격리·도구 차단·blind 라벨·JSONL). 단일턴 + **다중턴 에이전트 루프** 둘 다.
- [x] **채점기 프로토타입 3축** — 모두 judge-free/객관, stdlib only:
  - ⑤ 비압축성(gzip 압축률 + 비반복 + 알맹이밀도)
  - ⑥ 아첨저항(behavioral diff, 부정어 인식)
  - **⑨ 도구 오케스트레이션(mock-tool)** — 가짜 툴 + 각본 시뮬레이터 + 불변식 채점. 게이트 준수·순서·멈춤·환각·오류회복을 측정. 실제 sandbox 없어 환경 익스플로잇 불가.
- [x] **레이더 리포트** — 막대 차트 + N축 다각형 SVG(≥3축) + HTML 점수표 자동 생성
- [ ] 8축 최종 확정 + ⑨를 ①②⑥에 분배할지(직교성 검토) — 현재 3:5로 진행 중
- [ ] 나머지 축 채점기 (실행·논리 / 지시장악 / 사실·견고 / 창의·발산 / 청중적응 / 라포)
- [ ] 상대/백분위 정규화(인간 코퍼스·다모델 확장 시)

## 구현 (코드)
```
bench/
├── config.py     # 모델·effort·경로·격리 정책
├── client.py     # claude -p 격리 래퍼(구독 호출, ANTHROPIC_API_KEY 없을 때)
├── probes.py     # 단일턴 축 입력(아첨은 수치 난수화 = 차등 퍼징)
├── scenarios.py  # mock-tool 시나리오 생성 + 호출 규약 + 각본 시뮬레이터
├── runner.py     # probe×모델×N 실행(단일턴+다중턴 루프) → results/raw/<run_id>/*.jsonl
├── axes/         # 축별 채점기(density, sycophancy, tooluse) + 레지스트리
├── report.py     # 집계 + 레이더 SVG/HTML
└── cli.py        # python -m bench {run|report|score|smoke}
```

### 사용법
```bash
python3 -m bench smoke                 # haiku 3콜 스모크(파이프라인 확인)
python3 -m bench run                   # 기본: 4.8 vs 4.6, 두 축 전부, high effort
python3 -m bench run --limit 4 --effort high --workers 3
python3 -m bench score                 # 최신 run 점수만 JSON 출력
python3 -m bench report --run results/raw/<run_id>   # 리포트 재생성
```
- 모델 호출은 **구독 기반**(`ANTHROPIC_API_KEY` 미설정 시 `claude -p`가 구독에서 차감).
- 호출 가능한 Opus 사다리: 4.0 · 4.1 · 4.5 · 4.6 · 4.7 · 4.8 (`--models`로 지정).
- 새 축 추가 = `probes.py`에 probe + `axes/<name>.py`에 `score()` + 레지스트리 등록.

## Mindmatch — 모델 동시 플레이 아레나

여러 모델에게 같은 게임을 **한꺼번에** 시키고, 그 과정을 대시보드에서 실시간으로 지켜보는
제품이다. 현재 게임은 꼬맨틀(`ko-semantle`, Qwen3 4B 로컬 임베딩 + 고정 기준 어휘) 하나.

- 모든 모델이 에피소드별 **동일 seed(동일 정답)**를 받아 공정 비교. 대결 프레이밍 없음.
- 매 턴 즉시 저장(중단 강건) + 대시보드 2초 갱신: 현재 추측·유사도·순위·모델 응답 원문
  ("모델 공개 출력" — 비공개 chain-of-thought가 아님)이 진행 중에 바로 보인다.
- 저장·조회 단위는 모델: `results/arena/<run_id>/models/<모델>/` (모델 카드 클릭 → 과거 런).
- 채점은 LLM 심판 없이 고정 오라클로만. 완료 시 같은 오라클로 자동 재생 검증. 정답은
  에피소드가 끝나기 전엔 파일에도 API에도 노출되지 않는다.

```bash
python3 -m bench arena serve --port 8777          # Mindmatch 대시보드
python3 -m bench arena run --game ko-semantle \
  --models claude-haiku-4-5 codex-5.6-luna \
  --episodes 3 --effort low                        # 화면의 [새 플레이 시작]과 동일
python3 -m bench arena verify results/arena/<run_id>
```

## 다음 단계
1. 8축 확정 (`01-design.md`의 열린 결정 — 하드:스타일 3:5 vs 4:4)
2. 우선순위 축부터 채점기 확장(청중적응⑦이 "자폐 3증상"의 남은 하나)

## 문서
- `docs/01-design.md` — 축 설계, Tier 1 레이더 8축 + Tier 2 진단표
- `docs/02-methodology.md` — 채점 원칙, 축별 측정 스펙, 자폐 묶음 측정안
- `docs/03-research.md` — 딥리서치/Gemini 자문 결론 + 인용
