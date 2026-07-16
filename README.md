# llm-bench

여러 LLM을 같은 조건에서 실행하고 결과를 비교하는 개인용 벤치마크 도구입니다.

두 가지 방식으로 모델을 평가합니다.

- **축별 벤치마크**: 지시 이행, 도구 사용, 출력 밀도, 아첨 저항, 청중 적응, 창의성을 측정합니다.
- **Mindmatch**: 여러 모델이 같은 문제를 풀게 하고 진행 과정과 결과를 비교합니다.

## 주요 기능

- Claude Code, Codex CLI, Gemini 게이트웨이를 통한 모델 실행
- 모델별 effort 설정과 반복 실행
- 원시 응답 JSONL 저장
- 점수 집계와 HTML/SVG 리포트 생성
- 동일 seed를 사용한 모델 간 비교
- 저장된 게임 실행의 재생 검증
- 실시간 진행 상황을 보여주는 로컬 대시보드

## Mindmatch 게임

| 게임 | 설명 |
|---|---|
| `ko-semantle` | 의미 유사도를 이용해 정답 단어 찾기 |
| `ko-minefield` | 의미 지뢰를 피하며 정답 단어 찾기 |
| `ko-maze` | 제한된 정보로 미로 탐색하기 |
| `ko-rulelab` | 입출력 예시로 숨은 규칙 추론하기 |

## 설치

Python 3.9 이상이 필요합니다.

```bash
git clone https://github.com/chenjingdev/llm-bench.git
cd llm-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

모델 실행에는 인증된 Claude Code 또는 Codex CLI가 필요합니다. `ko-semantle`과 `ko-minefield`는 로컬 Ollama 임베딩 모델을 사용합니다.

## 사용법

### 축별 벤치마크

```bash
# 최소 실행으로 환경 확인
python3 -m bench smoke

# 모델과 평가 축 지정
python3 -m bench run \
  --models claude-haiku-4-5 codex-5.4 \
  --axes instruction tooluse density \
  --effort low

# 저장된 결과에서 점수 또는 리포트 생성
python3 -m bench score --run results/raw/<run_id>
python3 -m bench report --run results/raw/<run_id>
```

### Mindmatch

```bash
# 대시보드 실행
python3 -m bench arena serve --port 8777

# 같은 게임을 여러 모델로 실행
python3 -m bench arena run \
  --game ko-rulelab \
  --models claude-haiku-4-5 codex-5.4 \
  --episodes 3 \
  --effort low

# 저장된 실행 재검증
python3 -m bench arena verify results/arena/<run_id>
```

## 결과 파일

```text
results/
├── raw/       # 축별 벤치마크 원시 응답
├── reports/   # 집계 결과와 리포트
└── arena/     # Mindmatch 실행 기록
```

## 테스트

```bash
python3 -m pytest -q
```

설계 문서와 측정 방식은 [`docs/`](docs/)에 정리되어 있습니다.
