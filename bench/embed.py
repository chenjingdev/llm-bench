"""로컬 임베딩 클라이언트 — ollama nomic-embed-text (API 키 불필요).

창의·발산 축의 '의미 거리(semantic divergence)' 측정에 사용.
채점 시점에만 호출(모델 응답 텍스트를 임베딩) → 벤치 실행과 분리, 과금 없음.
"""

from __future__ import annotations

import json
import math
import urllib.request

OLLAMA = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
# nomic-embed-text v1.5는 태스크 프리픽스 권장. 또래 항목 간 유사도/군집엔 clustering.
_PREFIX = "clustering: "

_cache: dict[str, list[float]] = {}


def available() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def embed(texts: list[str], prefix: bool = True) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. 캐시 사용."""
    out: list[list[float] | None] = [None] * len(texts)
    todo, todo_idx = [], []
    for i, t in enumerate(texts):
        key = (_PREFIX if prefix else "") + t
        if key in _cache:
            out[i] = _cache[key]
        else:
            todo.append((_PREFIX if prefix else "") + t)
            todo_idx.append(i)
    if todo:
        body = json.dumps({"model": MODEL, "input": todo}).encode()
        req = urllib.request.Request(OLLAMA, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        embs = data["embeddings"]
        for j, idx in enumerate(todo_idx):
            v = embs[j]
            out[idx] = v
            _cache[texts[idx] if not prefix else _PREFIX + texts[idx]] = v
    return out  # type: ignore


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mean_pairwise_distance(vecs: list[list[float]]) -> float:
    """평균 쌍별 코사인 거리(1 - cos). 발산 클수록 ↑."""
    n = len(vecs)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1.0 - cosine(vecs[i], vecs[j])
            cnt += 1
    return tot / cnt if cnt else 0.0
