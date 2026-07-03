from bench import probes


def test_creativity_probes_four_subtypes():
    ps = probes.creativity_probes(seed=0)
    subs = {p.meta["subtype"] for p in ps}
    assert subs == {"tech", "copy", "humor", "metaphor"}


def test_creativity_probes_deterministic():
    a = [p.prompt for p in probes.creativity_probes(seed=0)]
    b = [p.prompt for p in probes.creativity_probes(seed=0)]
    assert a == b


def test_creativity_probes_carry_prompt_in_meta():
    ps = probes.creativity_probes(seed=1)
    assert all(p.meta.get("prompt") == p.prompt for p in ps)


def test_registry_has_creativity():
    ps = probes.build("creativity", seed=0)
    assert len(ps) >= 4


# --- 형식 계약(fmt) 레이어 ---
def test_creativity_probes_fmt_report_appends_contract():
    ps = probes.creativity_probes(seed=0, fmt="report")
    for p in ps:
        assert "FORMAT CONTRACT" in p.prompt
        assert "FORMAT CONTRACT" not in p.meta["prompt"]   # 온토픽 앵커는 과제 의미만
        assert p.prompt.startswith(p.meta["prompt"])
        assert p.meta["fmt"] == "report"


def test_creativity_probes_fmt_none_unchanged():
    plain = probes.creativity_probes(seed=0)
    explicit = probes.creativity_probes(seed=0, fmt=None)
    assert [p.prompt for p in plain] == [p.prompt for p in explicit]
    assert all("FORMAT CONTRACT" not in p.prompt for p in plain)


def test_build_passes_fmt_through_and_other_axes_tolerate_it():
    ps = probes.build("creativity", seed=0, fmt="report")
    assert all("FORMAT CONTRACT" in p.prompt for p in ps)
    # 타 축 빌더는 fmt를 무시(**_)하고 정상 동작해야 러너 관통이 안전
    assert probes.build("density", seed=0, fmt="report")
