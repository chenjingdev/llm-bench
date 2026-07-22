"""기준어휘 벡터 디스크 캐시 + 워밍업 핑 테스트 — 실 ollama 없이 돈다.

_post_embed를 가짜로 대체해 임베딩 호출 횟수를 세고, config.EMBED_CACHE_DIR을
tmp로 돌려 실제 캐시를 오염시키지 않는다. 캐시 로직만 볼 땐 warm=False로 워밍업을
끈다(워밍업 네트워크와 분리).
"""

import time

import json

from bench import config, embed
from bench.games import semantle as sm


def _fake_post(calls, dim=3):
    def post(body, timeout=120):
        calls["n"] += 1
        calls["last_timeout"] = timeout
        n = len(json.loads(body.decode())["input"])
        return {"embeddings": [[0.1 * (i + 1)] * dim for i in range(n)]}
    return post


# ----------------------------------------------------------------------
# 디스크 캐시: 미스 → 계산+기록, 적중 → 임베딩 호출 0, 손상 → 재계산 폴백
# ----------------------------------------------------------------------
def test_vocab_cache_miss_then_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_CACHE_DIR", tmp_path)
    embed._cache.clear()
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls))

    words = ["가", "나", "다"]
    v1 = embed.embed_vocab_cached(words, prefix=False, model="probe", warm=False)
    assert calls["n"] == 1                        # 미스: 1회 계산
    assert len(v1) == 3 and all(len(v) == 3 for v in v1)
    assert len(list(tmp_path.glob("*.json"))) == 1   # 캐시 파일 기록됨

    # 프로세스 내 캐시까지 비워도 디스크 적중이면 임베딩 호출 0.
    embed._cache.clear()
    calls["n"] = 0
    v2 = embed.embed_vocab_cached(words, prefix=False, model="probe", warm=False)
    assert calls["n"] == 0                        # 적중: 임베딩 호출 없음
    assert v2 == v1


def test_vocab_cache_key_separates_model_prefix_and_words(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_CACHE_DIR", tmp_path)
    embed._cache.clear()
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls))

    kw = dict(warm=False)
    embed.embed_vocab_cached(["가", "나"], prefix=False, model="probe", **kw)
    embed.embed_vocab_cached(["가", "나"], prefix=True, model="probe", **kw)    # prefix 다름
    embed.embed_vocab_cached(["가", "나"], prefix=False, model="other", **kw)   # 모델 다름
    embed.embed_vocab_cached(["가", "다"], prefix=False, model="probe", **kw)   # 어휘 다름
    assert calls["n"] == 4                        # 넷 다 별개 키 → 전부 미스
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_vocab_cache_corrupt_recomputes_and_rewrites(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_CACHE_DIR", tmp_path)
    embed._cache.clear()
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls))

    words = ["학교", "의사"]
    embed.embed_vocab_cached(words, prefix=False, model="probe", warm=False)   # 기록
    path = list(tmp_path.glob("*.json"))[0]

    for corrupt in ("{ broken json",                                    # 파싱 실패
                    json.dumps({"vectors": [[0.5, 0.6]]}),              # 개수 불일치(1≠2)
                    json.dumps({"vectors": [[0.5, 0.6], [0.5]]})):      # 차원 불일치
        path.write_text(corrupt, encoding="utf-8")
        embed._cache.clear()
        calls["n"] = 0
        v = embed.embed_vocab_cached(words, prefix=False, model="probe", warm=False)
        assert calls["n"] == 1                    # 손상 → 재계산
        assert len(v) == 2 and all(len(x) == 3 for x in v)
        # 재기록되어 다음엔 적중.
        embed._cache.clear()
        calls["n"] = 0
        embed.embed_vocab_cached(words, prefix=False, model="probe", warm=False)
        assert calls["n"] == 0


def test_vocab_cache_read_failure_is_silent(monkeypatch, tmp_path):
    # 읽기 실패(경로가 디렉토리 등) → 조용히 재계산, 예외 전파 없음.
    monkeypatch.setattr(config, "EMBED_CACHE_DIR", tmp_path)
    embed._cache.clear()
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls))
    words = ["바다"]
    path = embed._embed_cache_path("probe", False, words)
    path.mkdir(parents=True)                      # 파일 자리에 디렉토리 → read 실패 유발
    v = embed.embed_vocab_cached(words, prefix=False, model="probe", warm=False)
    assert calls["n"] == 1 and len(v) == 1        # 예외 없이 재계산


# ----------------------------------------------------------------------
# 워밍업: 긴 타임아웃 단건 호출 + 미스는 동기, 적중은 백그라운드 예열
# ----------------------------------------------------------------------
def test_warmup_uses_long_timeout(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls, dim=1))
    embed.warmup(model="probe", prefix=False)
    assert calls["n"] == 1
    assert calls["last_timeout"] == embed.WARMUP_TIMEOUT
    assert embed.WARMUP_TIMEOUT >= 600            # 콜드 로딩 흡수용


def test_cache_miss_warms_up_synchronously(monkeypatch, tmp_path):
    # 캐시 미스: 어휘 임베딩(120s) 전에 워밍업(600s)을 동기로 태워 콜드 로딩을 흡수한다.
    monkeypatch.setattr(config, "EMBED_CACHE_DIR", tmp_path)
    embed._cache.clear()
    seq = []
    monkeypatch.setattr(embed, "warmup",
                        lambda *a, **k: seq.append("warmup"))
    calls = {"n": 0}

    def post(body, timeout=120):
        calls["n"] += 1
        seq.append("embed")
        n = len(json.loads(body.decode())["input"])
        return {"embeddings": [[0.2] * 3 for _ in range(n)]}

    monkeypatch.setattr(embed, "_post_embed", post)
    embed.embed_vocab_cached(["가", "나"], prefix=False, model="probe")
    assert seq == ["warmup", "embed"]             # 워밍업 먼저(동기), 그 다음 임베딩
    assert calls["n"] == 1


def test_oracle_build_warms_up_and_disk_caches_vocab(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_CACHE_DIR", tmp_path)
    embed._cache.clear()
    warm = {"n": 0}
    monkeypatch.setattr(embed, "warmup",
                        lambda *a, **k: warm.__setitem__("n", warm["n"] + 1))
    monkeypatch.setattr(sm.embed, "model_info",
                        lambda m: {"name": m, "digest": "sha256:x"})
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls, dim=4))

    # 첫 빌드(캐시 미스): 워밍업 동기 1회 + 어휘 임베딩 1회 + 디스크 캐시 기록.
    oracle = sm.EmbeddingOracle(words=("가", "나", "다"), model="probe")
    assert warm["n"] == 1
    assert calls["n"] == 1
    assert len(oracle._vectors) == 3
    assert len(list(tmp_path.glob("*.json"))) == 1

    # 두 번째 빌드(캐시 적중): 어휘 재임베딩 0(빌드 즉시 반환), 워밍업은 백그라운드로 수행.
    warm["n"] = 0
    calls["n"] = 0
    embed._cache.clear()
    sm.EmbeddingOracle(words=("가", "나", "다"), model="probe")
    assert calls["n"] == 0                        # 어휘 재임베딩 없음(캐시 적중)
    for _ in range(200):                          # 백그라운드 예열 완료 대기(최대 ~2s)
        if warm["n"] >= 1:
            break
        time.sleep(0.01)
    assert warm["n"] == 1                          # 적중이어도 플레이 대비 예열은 수행


# ----------------------------------------------------------------------
# 청크 분할: 대량 입력을 EMBED_CHUNK 단위로 나눠 순차 POST(순서 보존)
# ----------------------------------------------------------------------
def test_embed_chunks_large_input_preserving_order(monkeypatch):
    import math
    embed._cache.clear()
    chunks = []

    def fake_post(body, timeout=120):
        inp = json.loads(body.decode())["input"]
        chunks.append(len(inp))
        return {"embeddings": [[float(int(t[1:]))] for t in inp]}   # 인덱스 인코딩

    monkeypatch.setattr(embed, "_post_embed", fake_post)
    n = embed.EMBED_CHUNK * 2 + 5                 # 청크 경계를 여러 번 넘김
    texts = [f"w{i}" for i in range(n)]
    out = embed.embed(texts, prefix=False, model="probe")

    # 청크 수 = ceil(n/CHUNK), 각 청크 ≤ CHUNK, 총합 == n(단일 거대 POST 아님)
    assert len(chunks) == math.ceil(n / embed.EMBED_CHUNK)
    assert all(c <= embed.EMBED_CHUNK for c in chunks) and max(chunks) == embed.EMBED_CHUNK
    assert sum(chunks) == n
    # 반환 순서 = 입력 순서(청크 경계 넘어서도 정확히 매핑)
    assert out == [[float(i)] for i in range(n)]


def test_embed_small_input_single_post(monkeypatch):
    # 소량 입력(기존 경로) 불변: 청크 1개(단일 POST)로 처리.
    embed._cache.clear()
    calls = {"n": 0}
    monkeypatch.setattr(embed, "_post_embed", _fake_post(calls))
    out = embed.embed(["가", "나", "다"], prefix=False, model="probe")
    assert calls["n"] == 1 and len(out) == 3
