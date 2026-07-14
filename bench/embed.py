"""로컬 임베딩 클라이언트 — ollama nomic-embed-text (API 키 불필요).

창의·발산 축의 '의미 거리(semantic divergence)' 측정에 사용.
채점 시점에만 호출(모델 응답 텍스트를 임베딩) → 벤치 실행과 분리, 과금 없음.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request

OLLAMA_BASE = "http://localhost:11434"
OLLAMA = f"{OLLAMA_BASE}/api/embed"
MODEL = "nomic-embed-text"
# nomic-embed-text v1.5는 태스크 프리픽스 권장. 또래 항목 간 유사도/군집엔 clustering.
_PREFIX = "clustering: "

# 임베딩 호출 재시도 백오프(초). 최초 시도 + len회 재시도. 일시 포화·모델 리로드로
# 인한 socket.timeout 등이 게임 턴을 통째로 죽이지 않게 흡수한다(소진 시에만 예외).
EMBED_RETRY_BACKOFF = (2.0, 5.0)

# 캐시 키는 (model, text) — 모델별로 벡터가 다르므로 분리한다.
_cache: dict[tuple[str, str], list[float]] = {}


def available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _post_embed(body: bytes) -> dict:
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _post_embed_with_retry(body: bytes) -> dict:
    """임베딩 POST를 재시도. socket.timeout/URLError 등 일시 장애를 흡수한다."""
    last: BaseException | None = None
    for attempt in range(len(EMBED_RETRY_BACKOFF) + 1):
        try:
            return _post_embed(body)
        except OSError as exc:  # socket.timeout·URLError 모두 OSError 계열
            last = exc
            if attempt < len(EMBED_RETRY_BACKOFF):
                time.sleep(EMBED_RETRY_BACKOFF[attempt])
    raise last  # type: ignore[misc]


def embed(texts: list[str], prefix: bool = True, *, model: str = MODEL) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. (model, text) 단위 캐시."""
    out: list[list[float] | None] = [None] * len(texts)
    todo, todo_idx = [], []
    for i, t in enumerate(texts):
        text = (_PREFIX if prefix else "") + t
        key = (model, text)
        if key in _cache:
            out[i] = _cache[key]
        else:
            todo.append(text)
            todo_idx.append(i)
    if todo:
        body = json.dumps({"model": model, "input": todo}).encode()
        data = _post_embed_with_retry(body)
        embs = data["embeddings"]
        for j, idx in enumerate(todo_idx):
            v = embs[j]
            out[idx] = v
            _cache[(model, todo[j])] = v
    return out  # type: ignore


def model_info(model: str = MODEL) -> dict:
    """ollama /api/tags에서 모델의 {name, digest} 조회.

    정확일치 우선, 없으면 접두일치. 조회 실패 시 {name: model, digest: "unknown"}.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
    except Exception:
        return {"name": model, "digest": "unknown"}
    entries = data.get("models", []) if isinstance(data, dict) else []
    # 정확일치
    for entry in entries:
        if entry.get("name") == model:
            return {"name": entry.get("name", model),
                    "digest": entry.get("digest", "unknown")}
    # 접두일치(예: "qwen3-embedding" 요청 → "qwen3-embedding:4b" 태그)
    for entry in entries:
        name = entry.get("name", "")
        if name.startswith(model):
            return {"name": name, "digest": entry.get("digest", "unknown")}
    return {"name": model, "digest": "unknown"}


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
        return max(kdist[j], d_ij, 1e-9)

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
