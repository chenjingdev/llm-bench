# GOAL — ChatGPT 웹에서 아이디어 발산 벤치 자동 수집

너는 브라우저 자동화로 **ChatGPT 웹(https://chatgpt.com)** 에서 4개의 브레인스토밍 프롬프트를
돌리고, 모델의 답을 **그대로** 긁어서 `divergence_web.jsonl` 파일 하나로 저장한다.
아이디어를 해석/요약/번역하지 말고 **원문 verbatim** 으로만 저장한다.

## 전제
- 사용자는 이미 chatgpt.com 에 로그인되어 있다. 로그인 벽/캡차/차단이 뜨면 **즉시 멈추고**
  어느 단계에서 막혔는지 보고한다. (로그인·결제·설정 변경은 절대 하지 않는다.)
- 작업은 **읽기 전용**: 새 대화 생성과 응답 읽기만 한다. 채팅 삭제·공유·share·설정변경·
  Upgrade 클릭 금지.

## 절차 (프롬프트 4개를 각각 독립적으로)
각 프롬프트마다 아래를 반복한다. **반드시 매번 새 대화(빈 채팅)에서** 시작한다 —
이전 답이 맥락에 남으면 결과가 오염된다.

1. `https://chatgpt.com` 에서 **New chat**(새 채팅)을 연다. 이전 대화 맥락이 없어야 한다.
2. 화면의 모델 선택기에 표시된 **현재 모델 이름을 그대로 기록**한다 (예: "GPT-5.1").
   네 가지 프롬프트 모두 **같은 모델**로 돌린다.
3. 아래 해당 프롬프트 텍스트를 입력창에 **그대로** 붙여넣고 전송한다.
4. **응답이 완전히 끝날 때까지 기다린다.** 답이 길다(12개 항목). 스트리밍/중지 버튼이
   사라지고 전송 버튼이 다시 활성화되면 완료로 본다.
5. 어시스턴트의 **마지막 답변 전체 텍스트를 verbatim 으로 추출**한다 (번호 목록 1–12).
6. 점검: 1번부터 12번까지 번호 항목이 대략 다 있는지 확인한다. 잘려서 12개 미만이면
   "continue" 를 보내거나 한 번 다시 생성해서 12개를 채운다. 그래도 안 되면 받은 만큼
   저장하되 그 줄의 `ok` 를 `false` 로 둔다.

## 4개 프롬프트

### domain = rag
```
I'm researching new approaches to retrieval-augmented generation (RAG). Beyond the standard pipeline (chunk → embed → vector search → rerank → stuff into context), brainstorm 12 genuinely novel RAG ideas or architectures worth exploring. Give original, non-obvious directions — not the textbook ones. Number them 1–12, one idea per line.
```

### domain = uncertainty
```
Brainstorm 12 unconventional, non-obvious ways to make an LLM reliably recognize and signal when it doesn't actually know something. Avoid the obvious ('add confidence scores', 'use RAG', 'fine-tune on refusals'). I want fresh, original mechanisms. Number them 1–12, one per line.
```

### domain = agent_memory
```
Brainstorm 12 original ideas for giving an AI coding agent useful long-term memory across many sessions. Go beyond 'store embeddings in a vector DB'. I want novel, non-obvious mechanisms — even speculative ones. Number them 1–12, one per line.
```

### domain = llm_eval
```
Brainstorm 12 genuinely novel ways to measure whether one LLM is 'better' for a specific person's real workflow, beyond standard benchmarks and A/B preference votes. Original, non-obvious ideas only — speculative is fine. Number them 1–12, one per line.
```

## 최종 출력 파일 — `divergence_web.jsonl`
- **JSON Lines**: 한 줄에 JSON 객체 하나, **총 4줄**(도메인당 1줄). UTF-8.
- 각 줄 스키마(필드명·구조 정확히 이대로):

```json
{"axis":"divergence","probe_id":"div_rag","model":"gpt-web","repeat":0,"prompt":"<붙여넣은 프롬프트 원문>","text":"<모델 답변 전체 verbatim>","meta":{"domain":"rag","ui_model":"<2번에서 기록한 모델명>","source":"chatgpt-web"},"ok":true}
```

규칙:
- `probe_id` 는 `div_<domain>` (`div_rag`, `div_uncertainty`, `div_agent_memory`, `div_llm_eval`).
- `meta.domain` 은 정확히 `rag` / `uncertainty` / `agent_memory` / `llm_eval` 중 하나.
- `model` 은 `gpt-web` 고정. 실제 UI 모델명은 `meta.ui_model` 에 넣는다.
- `text` 는 모델 답변의 **원문 그대로**. 줄바꿈은 JSON 으로 `\n` 이스케이프해서 한 줄 안에 보존.
  번호 목록(`1. ...`) 형식을 그대로 유지(파서가 그걸로 항목을 자른다).
- 아이디어를 임의로 자르거나 줄이지 말 것. 12개 다 넣는다.

## 완료 보고
- `divergence_web.jsonl` 의 **절대 경로**, 줄 수(4여야 함), 각 줄의 `meta.domain` 과
  `text` 글자 수를 표로 보고한다.
- 중간에 막힌 도메인이 있으면 어느 단계(1~6)에서 왜 막혔는지 적는다.
