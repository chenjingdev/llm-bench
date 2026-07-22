"""로컬 임베딩 클라이언트 — ollama nomic-embed-text (API 키 불필요).

창의·발산 축의 '의미 거리(semantic divergence)' 측정에 사용.
채점 시점에만 호출(모델 응답 텍스트를 임베딩) → 벤치 실행과 분리, 과금 없음.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from . import config

OLLAMA_BASE = "http://localhost:11434"
OLLAMA = f"{OLLAMA_BASE}/api/embed"
MODEL = "nomic-embed-text"
# nomic-embed-text v1.5는 태스크 프리픽스 권장. 또래 항목 간 유사도/군집엔 clustering.
_PREFIX = "clustering: "

# 임베딩 호출 재시도 백오프(초). 최초 시도 + len회 재시도. 일시 포화·모델 리로드로
# 인한 socket.timeout 등이 게임 턴을 통째로 죽이지 않게 흡수한다(소진 시에만 예외).
EMBED_RETRY_BACKOFF = (2.0, 5.0)

# 대량 입력(어휘 5,000개 등)을 단일 POST로 보내면 ollama가 순차 계산(~180ms/단어)하는
# 동안 클라이언트 타임아웃(120s×재시도)을 소진해 socket.timeout으로 실패한다(실측: 2.0.0
# 오라클 빌드 불가). 청크 단위로 나눠 순차 POST하면 청크당 타임아웃 안에 끝난다(128×0.18s
# ≈ 23s < 120s). 전체 소요(~15분)는 어휘 빌드 1회에 한하며 이후는 디스크 캐시 즉답.
EMBED_CHUNK = 128

# 캐시 키는 (model, text) — 모델별로 벡터가 다르므로 분리한다.
_cache: dict[tuple[str, str], list[float]] = {}


def available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False


# 콜드 로딩 흡수용 긴 타임아웃 — 12GB 임베딩 모델 GPU 상주에 실측 수 분까지 걸린다
# (콜드 build_game이 120s×재시도를 넘겨 367초 후 timeout 실패한 사례). 워밍업만 이
# 타임아웃을 쓰고, 일반 턴 임베딩 경로(120s)는 그대로 둔다.
WARMUP_TIMEOUT = 600


def _post_embed(body: bytes, timeout: int = 120) -> dict:
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_embed_with_retry(body: bytes, timeout: int = 120) -> dict:
    """임베딩 POST를 재시도. socket.timeout/URLError/HTTPError 등 일시 장애를 흡수한다."""
    last: BaseException | None = None
    for attempt in range(len(EMBED_RETRY_BACKOFF) + 1):
        try:
            return _post_embed(body, timeout=timeout)
        except OSError as exc:  # socket.timeout·URLError·HTTPError 모두 OSError 계열
            last = exc
            if attempt < len(EMBED_RETRY_BACKOFF):
                time.sleep(EMBED_RETRY_BACKOFF[attempt])
    raise last  # type: ignore[misc]


def warmup(model: str = MODEL, prefix: bool = True) -> None:
    """모델 로딩 흡수 — 어휘 일괄 임베딩 전에 단건을 긴 타임아웃(WARMUP_TIMEOUT)으로
    호출해 콜드 스타트(모델 GPU 상주)를 흡수한다. 콜드 로딩 직후 일시 HTTP 400도
    관찰됐으므로 _post_embed_with_retry의 재시도가 그 첫 실호출 흔들림을 흡수한다."""
    text = (_PREFIX if prefix else "") + "워밍업"
    body = json.dumps({"model": model, "input": [text]}).encode()
    _post_embed_with_retry(body, timeout=WARMUP_TIMEOUT)


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
        # 청크 분할 순차 POST — 청크별 재시도, 결과를 입력 순서대로 이어붙인다.
        embs: list[list[float]] = []
        total = len(todo)
        for start in range(0, total, EMBED_CHUNK):
            chunk = todo[start:start + EMBED_CHUNK]
            body = json.dumps({"model": model, "input": chunk}).encode()
            data = _post_embed_with_retry(body)
            embs.extend(data["embeddings"])
            if total > EMBED_CHUNK:   # 대량(어휘 빌드) 경로만 진행 로그(~15분 관찰용)
                print(f"[embed] {min(start + EMBED_CHUNK, total)}/{total} ({model})",
                      file=sys.stderr, flush=True)
        for j, idx in enumerate(todo_idx):
            v = embs[j]
            out[idx] = v
            _cache[(model, todo[j])] = v
    return out  # type: ignore


# ----------------------------------------------------------------------
# 기준어휘 벡터 디스크 캐시 — 웜 build_game의 어휘 재임베딩(실측 ~27초)을 제거한다.
# 순수 메모이제이션: 오라클 metadata(모델·prefix·어휘 해시)·measurement_key 불변.
# 오히려 어휘 벡터를 디스크에 고정해 임베딩 코배칭 노이즈(동시 요청 시 같은 입력이
# ~4.3e-4 흔들리는 실측)를 없애 재생 검증 재현성을 높인다. 손상·불일치·읽기 실패는
# 조용히 무시하고 재계산·재기록한다. arena를 임포트하지 않는 자체 원자 쓰기를 쓴다.
# ----------------------------------------------------------------------
def _embed_cache_path(model: str, prefix: bool, words: list[str]) -> Path:
    key = json.dumps({"model": model, "prefix": bool(prefix), "words": list(words)},
                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(config.EMBED_CACHE_DIR) / f"{digest}.json"


def _write_json_atomic(path: Path, data) -> None:
    """임시파일 쓰고 os.replace로 원자 교체(부분쓰기 노출 차단)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _load_vocab_cache(path: Path, expected_count: int) -> list[list[float]] | None:
    """캐시 로드 + 정합성 검증(벡터 개수=어휘 수, 차원 일치). 불일치/손상 → None."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):        # 읽기 실패·JSON 파싱 실패 모두 포함
        return None
    if not isinstance(data, dict):
        return None
    vecs = data.get("vectors")
    if not isinstance(vecs, list) or len(vecs) != expected_count:
        return None
    dims = set()
    for v in vecs:
        if not isinstance(v, list) or not v:
            return None
        dims.add(len(v))
    if expected_count and len(dims) != 1:  # 차원 불일치(빈 어휘는 검증 생략)
        return None
    return vecs


def _store_vocab_cache(path: Path, model: str, prefix: bool,
                       words: list[str], vecs: list[list[float]]) -> None:
    digest = hashlib.sha256(
        json.dumps(list(words), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "model": model,
        "prefix": bool(prefix),
        "vocab_digest": "sha256:" + digest,
        "count": len(words),
        "dim": len(vecs[0]) if vecs else 0,
        "vectors": vecs,
    }
    try:
        _write_json_atomic(path, payload)
    except OSError:
        pass   # 캐시 기록 실패는 치명 아님 — 다음 빌드가 재계산한다


def _warmup_quiet(model: str, prefix: bool) -> None:
    try:
        warmup(model, prefix)
    except Exception:
        pass   # 백그라운드 예열 실패는 무시 — 플레이 첫 턴이 재시도 정책으로 흡수한다


def embed_vocab_cached(words: list[str], prefix: bool = True, *,
                       model: str = MODEL, warm: bool = True) -> list[list[float]]:
    """기준어휘 임베딩을 디스크 캐시로 메모이즈한다(적중 시 임베딩 호출 없음).

    warm=True면 콜드 스타트를 흡수한다:
      - 캐시 미스: 어휘 임베딩(120s 타임아웃, 콜드 로딩이 이를 넘겨 실패한 사례) 전에
        워밍업을 '동기'로 태워 모델 로딩(실측 수 분)을 긴 타임아웃으로 흡수한다.
      - 캐시 적중: 워밍업을 '백그라운드'로 돌려 빌드를 즉시 반환한다(웜 재빌드 <1s).
        예열은 이후 플레이 턴이 콜드 스타트에 걸리지 않도록 준비만 한다.
    """
    words = list(words)
    path = _embed_cache_path(model, prefix, words)
    cached = _load_vocab_cache(path, len(words))
    if cached is not None:
        if warm:
            threading.Thread(target=_warmup_quiet, args=(model, prefix),
                             daemon=True).start()
        return cached
    if warm:
        warmup(model, prefix)     # 미스: 콜드 로딩 동기 흡수 후 어휘 임베딩
    vecs = embed(words, prefix=prefix, model=model)
    _store_vocab_cache(path, model, prefix, words, vecs)
    return vecs


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
