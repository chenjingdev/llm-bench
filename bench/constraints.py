"""축 ② 지시 장악 — 프로그램적 제약 검사기(judge-free, 100% 객관).

검증 가능한 제약을 여러 개 쌓아 모델이 동시에 몇 개를 지키는지 측정한다.
프런티어 모델은 '제약 저글링'에서 갈린다(카운팅·리포그램·아크로스틱·포맷).
제약 spec(type+params)을 레코드에 저장 → 채점 시 체커 재구성(서버 상태 불필요).
"""

from __future__ import annotations

import re

_WORDS = re.compile(r"[A-Za-z0-9']+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


# ---- 체커: (text, params) -> bool, 그리고 렌더: (params) -> 지시문 -------------
def _c_sentence_count(t, p):
    return len(_sentences(t)) == p["n"]


def _c_word_count(t, p):
    n = len(_WORDS.findall(t))
    return p["lo"] <= n <= p["hi"]


def _c_forbidden_letter(t, p):
    return p["letter"].lower() not in t.lower()


def _c_forbidden_word(t, p):
    return re.search(rf"\b{re.escape(p['word'])}\b", t, re.I) is None


def _c_required_word_count(t, p):
    return len(re.findall(rf"\b{re.escape(p['word'])}\b", t, re.I)) == p["k"]


def _c_all_lowercase(t, p):
    return not any(c.isupper() for c in t)


def _c_no_commas(t, p):
    return "," not in t


def _c_acrostic(t, p):
    ls = _lines(t)
    w = p["word"]
    if len(ls) != len(w):
        return False
    return "".join(l[0].lower() for l in ls if l) == w.lower()


def _c_each_line_period(t, p):
    ls = _lines(t)
    return len(ls) >= 1 and all(l.endswith(".") for l in ls)


def _c_numbered_list(t, p):
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\.\s", t, re.M)]
    return nums == list(range(1, p["n"] + 1))


def _r_sentence_count(p):
    return f"Write exactly {p['n']} sentences."


def _r_word_count(p):
    return f"Use between {p['lo']} and {p['hi']} words total."


def _r_forbidden_letter(p):
    return f"Do not use the letter '{p['letter']}' anywhere in your response."


def _r_forbidden_word(p):
    return f"Never use the word \"{p['word']}\"."


def _r_required_word_count(p):
    return f"Use the word \"{p['word']}\" exactly {p['k']} time(s)."


def _r_all_lowercase(p):
    return "Write entirely in lowercase (no capital letters at all)."


def _r_no_commas(p):
    return "Do not use any commas."


def _r_acrostic(p):
    return (f"Write exactly {len(p['word'])} lines; the first letter of each line, "
            f"read top to bottom, must spell \"{p['word'].upper()}\".")


def _r_each_line_period(p):
    return "Put each item on its own line, and end every line with a period."


def _r_numbered_list(p):
    return f"Format the answer as a numbered list with exactly {p['n']} items (1., 2., ...)."


REGISTRY = {
    "sentence_count": (_c_sentence_count, _r_sentence_count),
    "word_count": (_c_word_count, _r_word_count),
    "forbidden_letter": (_c_forbidden_letter, _r_forbidden_letter),
    "forbidden_word": (_c_forbidden_word, _r_forbidden_word),
    "required_word_count": (_c_required_word_count, _r_required_word_count),
    "all_lowercase": (_c_all_lowercase, _r_all_lowercase),
    "no_commas": (_c_no_commas, _r_no_commas),
    "acrostic": (_c_acrostic, _r_acrostic),
    "each_line_period": (_c_each_line_period, _r_each_line_period),
    "numbered_list": (_c_numbered_list, _r_numbered_list),
}


def check(ctype: str, text: str, params: dict) -> bool:
    return REGISTRY[ctype][0](text or "", params)


def render(ctype: str, params: dict) -> str:
    return REGISTRY[ctype][1](params)
