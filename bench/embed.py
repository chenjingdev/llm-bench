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


def knn_cosine_distances(vecs: list[list[float]], i: int, k: int) -> list[tuple[float, int]]:
    """vec i에서 코사인거리(1-cos) 가까운 순 (거리, 인덱스) k개(자기 제외)."""
    dists = [(1.0 - cosine(vecs[i], vecs[j]), j) for j in range(len(vecs)) if j != i]
    dists.sort(key=lambda t: t[0])
    return dists[:max(1, k)]


def lof(vecs: list[list[float]], k: int = 20) -> list[float]:
    """Local Outlier Factor(코사인거리 기반, 순수 파이썬).

    ~1 = 전형(이웃과 밀도 비슷), >1 = 국소적으로 외딴(독창).
    딥리서치 근거: 밀도비율이라 클러스터별 밀도차에 강건(sklearn 문서).
    """
    n = len(vecs)
    if n == 0:
        return []
    if n < 2:
        return [1.0] * n
    k = min(k, n - 1)
    neigh = [knn_cosine_distances(vecs, i, k) for i in range(n)]
    kdist = [neigh[i][-1][0] for i in range(n)]          # k-거리

    def reach_dist(i: int, j: int, d_ij: float) -> float:
        return max(kdist[j], d_ij)

    lrd = []                                              # 국소 도달밀도
    for i in range(n):
        s = sum(reach_dist(i, j, d) for d, j in neigh[i])
        lrd.append(len(neigh[i]) / s if s > 0 else float("inf"))

    scores = []
    for i in range(n):
        if lrd[i] == 0 or lrd[i] == float("inf"):
            scores.append(1.0)
            continue
        ratio = sum(lrd[j] for _, j in neigh[i]) / len(neigh[i]) / lrd[i]
        scores.append(ratio)
    return scores
