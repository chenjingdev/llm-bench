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
