#!/usr/bin/env python3
"""블라인드 코퍼스 탈식별: 모델 식별 단서가 되는 '끝맺음 습관 멘트' 제거.

대상(본문 손상 없는 꼬리 보일러플레이트만):
  1) 이어주기 제안형 클로저  — "원하면/필요하면/…알려주시면 …해드릴게요", "다음 답변에서 …",
     후속 정보요청 질문("지금 학년/부품 뭔지…?") → 트레일링 문장 통째 제거
  2) 요약 라벨            — "요약하면,/정리하면:/한 줄 요약:/결론:/결론적으로:" 등
     → 라벨(+볼드/콜론)만 제거하고 뒤 내용 문장은 그대로 보존

원본은 raw_outputs.raw.jsonl 로 백업(있으면 그걸 소스로 재처리 → 멱등).
--apply 없으면 감사 로그만 출력(드라이런).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results/raw/blind_pool_v2/raw_outputs.jsonl"
BAK = ROOT / "results/raw/blind_pool_v2/raw_outputs.raw.jsonl"

# 문장 경계: . ! ? (뒤 공백/EOL). 잘못 쪼개도 '마지막 문장이 클로저일 때만' 지우므로 안전.
_SENT = re.compile(r'[.!?]["\'\)\]]?(?=\s|$)')

_ASK = re.compile(r'(원하시?면|필요하시?면|궁금하시?면|알려\s*주시?면|알려주시?면|말씀해?\s*주시?면|'
                  r'있으면\s*(말씀|알려)|댓글|다음\s*답변에서)')
_OFFER = re.compile(r'(드릴게|드리겠|드려요|드릴 수|해\s*드리|정리해|짚어|잡아|좁혀|골라|'
                    r'줄게요|줄게|줄 수 있|주겠|판단|짜\s*드리|짜주|맞춤으로|맞춤형|콕\s*집어|'
                    r'딱\s*짚|말씀\s*드리|말씀드릴|로드맵으로)')
_Q_CTX = re.compile(r'(학년|어떤\s*상황|상황이세요|어떤\s*부품|무슨\s*부품|부품인지|뭔지|무엇을|'
                    r'어떤\s*.{0,8}(만들|쓰|하시|필요))')
# 작업수행형 제안(=널 위해 ~해줄게). '추천/권장합니다'(조언)는 제외되도록 작업 동사로 한정.
_TASK_OFFER = re.compile(r'(정리해|짚어|판단(해)?|짜\s*드리|짜\s*줄|골라|좁혀|잡아|기준표|로드맵|'
                         r'학습\s*계획|세팅).{0,10}(드리|드릴|줄게|줄 수|주겠)')
# "혹시 …있으면 알려주세요"형 정보요청 클로저
_LETMEKNOW = re.compile(r'(혹시|있으면|있다면|궁금하)')
_TELLME = re.compile(r'(알려\s*주세요|알려주세요|말씀해?\s*주세요|남겨\s*주세요|적어\s*주세요)')

_LBL = (r'요약하면|정리하면|정리하자면|한\s*줄\s*요약|한\s*줄로\s*말하면|한\s*줄로\s*정리하면|'
        r'결론만\s*말하면|결론부터\s*말하면|결론적으로')
# 볼드로 감싼 라벨일 때만 닫는 ** 흡수(여는 ** 있어야 닫는 ** 먹음 → 내용 볼드 안 건드림)
_RECAP_BOLD = re.compile(
    r'^\s*(?:>?\s*)(?:#{1,6}\s*)?'
    r'\*\*\s*(' + _LBL + r'|요약|결론)\s*[:：,]?\s*\*\*\s*[:：,]?\s*')
# 볼드 없는 강한 마커(다의어 아님): 콜론 없어도 제거. 트레일링 ** 흡수 안 함.
_RECAP_STRONG = re.compile(
    r'^\s*(?:>?\s*)(?:#{1,6}\s*)?(' + _LBL + r')(?=[\s:：,]|$)\s*[:：,]?\s*')
# 볼드 없는 약한 마커(요약본/결론은 오인 위험): 콜론 필수.
_RECAP_WEAK = re.compile(
    r'^\s*(?:>?\s*)(?:#{1,6}\s*)?(요약|결론)\s*[:：]\s*')
_HDR_ONLY = re.compile(r'^\s*(?:#{1,6}\s*)?(?:\*\*\s*)?(한\s*줄\s*요약|요약|결론)(?:\s*\*\*)?\s*$')


def is_solicit(sent: str) -> bool:
    s = sent.strip().strip('*>　 ').strip()
    if len(s) < 4:
        return False
    if re.search(r'다음\s*답변에서', s):
        return True
    if _ASK.search(s) and _OFFER.search(s):
        return True
    if re.search(r'더\s*구체적으로', s) and _OFFER.search(s):
        return True
    if s.endswith('?') and _Q_CTX.search(s):
        return True
    if _TASK_OFFER.search(s):
        return True
    if _LETMEKNOW.search(s) and _TELLME.search(s):
        return True
    return False


def split_sents(line: str):
    out, prev = [], 0
    for m in _SENT.finditer(line):
        out.append(line[prev:m.end()])
        prev = m.end()
    if prev < len(line):
        out.append(line[prev:])
    return out


def strip_trailing(text: str):
    removed = []
    text = text.rstrip()
    for _ in range(2):  # 최대 2개 꼬리 문장
        lines = text.split('\n')
        while lines and lines[-1].strip() in ('', '---', '***', '___', '> '):
            lines.pop()
        if not lines:
            break
        sents = split_sents(lines[-1])
        if sents and is_solicit(sents[-1]):
            removed.append(sents[-1].strip())
            rest = ''.join(sents[:-1]).rstrip()
            if rest.strip():
                lines[-1] = rest
            else:
                lines.pop()
                while lines and lines[-1].strip() in ('', '---', '***', '___', '> '):
                    lines.pop()
            text = '\n'.join(lines).rstrip()
        else:
            break
    return text, removed


def strip_recap(text: str):
    lines = text.split('\n')
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return text, None
    # 인라인: 마지막 줄이 라벨로 시작 + 뒤에 내용 → 라벨만 제거
    for rx in (_RECAP_BOLD, _RECAP_STRONG, _RECAP_WEAK):
        m = rx.match(lines[i])
        if m and lines[i][m.end():].strip():
            label = lines[i][:m.end()].strip()
            lines[i] = lines[i][m.end():]
            return '\n'.join(lines), label
    # 헤더 단독줄: 마지막 내용줄 바로 위(빈 줄 건너뜀)가 라벨 헤더면 그 줄 삭제
    j = i - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j >= 0 and _HDR_ONLY.match(lines[j]):
        label = lines[j].strip()
        del lines[j]
        return '\n'.join(lines), label
    return text, None


def clean(text: str):
    t, removed = strip_trailing(text)
    t, label = strip_recap(t)
    if label:
        removed.append("[라벨] " + label)
    return t.rstrip(), removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = BAK if BAK.exists() else SRC
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    changed = 0
    for o in rows:
        for fld in ("text", "a"):
            if not o.get(fld):
                continue
            new, removed = clean(o[fld])
            if new != o[fld].rstrip():
                changed += 1
                tag = f"{o['model']:16s} {o['probe']:11s} .{fld}"
                print(f"\n[{tag}]")
                for r in removed:
                    print(f"   ✂ {r[:150]}")
                print(f"   …끝 BEFORE: …{o[fld].rstrip()[-90:]!r}")
                print(f"   …끝 AFTER : …{new[-90:]!r}")
                o[fld] = new

    print(f"\n=== 변경 {changed}건 / 전체 필드 ===")
    if args.apply:
        if not BAK.exists():
            BAK.write_text(SRC.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[backup] {BAK.name}")
        SRC.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in rows) + "\n", encoding="utf-8")
        print(f"[apply] {SRC.name} 갱신")
    else:
        print("(드라이런 — --apply 로 반영)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
