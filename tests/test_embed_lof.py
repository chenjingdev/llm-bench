import math

from bench import embed


def test_knn_returns_k_sorted():
    vecs = [[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]]
    nn = embed.knn_cosine_distances(vecs, 0, 2)
    assert len(nn) == 2
    assert nn[0][0] <= nn[1][0]          # 거리 오름차순
    assert nn[0][1] in (1, 2)            # 가장 가까운 건 이웃 클러스터


def test_lof_flags_outlier_highest():
    # 세 개는 몰려있고 하나(마지막)는 외딴 점
    vecs = [[1, 0], [0.99, 0.01], [0.98, 0.02], [0, 1]]
    scores = embed.lof(vecs, k=2)
    assert len(scores) == 4
    assert scores[3] == max(scores)      # 외딴 점이 최고 LOF
    assert scores[3] > 1.0               # 아웃라이어 > 1


def test_lof_degenerate_small_pool():
    assert embed.lof([[1, 0]], k=5) == [1.0]     # 1개면 전형값
    assert embed.lof([], k=5) == []


def test_lof_duplicates_no_inf_and_low_novelty():
    # 완전 동일 벡터가 섞여도 inf/nan 없이 유한, 중복점은 아웃라이어보다 낮아야
    vecs = [[1, 0], [1, 0], [1, 0], [0, 1]]   # 앞 3개 동일(중복 클러스터), 마지막이 진짜 외딴점
    s = embed.lof(vecs, k=2)
    assert all(math.isfinite(x) for x in s)
    assert s[0] <= s[3]          # 중복점 < 진짜 아웃라이어
