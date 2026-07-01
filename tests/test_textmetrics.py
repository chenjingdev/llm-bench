from bench.axes import _textmetrics as tm


def test_compression_ratio_higher_for_repetitive():
    rep = "the cat sat. " * 30
    varied = ("Quantum entanglement links distant particles. "
              "Photosynthesis converts sunlight to sugar. "
              "Glaciers carve valleys over millennia. ")
    assert tm.compression_ratio(rep) > tm.compression_ratio(varied)


def test_compression_ratio_empty_is_one():
    assert tm.compression_ratio("") == 1.0


def test_long_ngram_repetition_detects_repeats():
    rep = "alpha beta gamma delta epsilon zeta eta theta " * 4
    uniq = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    assert tm.long_ngram_repetition(rep) > tm.long_ngram_repetition(uniq)


def test_self_bleu_higher_when_items_redundant():
    redundant = ["use it as a doorstop", "use it as a door stop",
                 "it can be a doorstop"]
    diverse = ["grind it into red pigment", "use it as a phone amplifier",
               "carve it into a chess set"]
    assert tm.self_bleu(redundant) > tm.self_bleu(diverse)


def test_mtld_zero_for_short_text():
    assert tm.mtld("too short here") == 0.0


def test_validity_gate_penalizes_repetition():
    rep = "idea one. idea one. idea one. idea one. idea one. " * 3
    items_rep = ["idea one"] * 12
    diverse_text = ("Compress embeddings with product quantization. "
                    "Route queries by learned difficulty. "
                    "Cache reasoning traces across sessions. "
                    "Detect drift via rolling perplexity bands. ")
    items_div = ["Compress embeddings with product quantization",
                 "Route queries by learned difficulty",
                 "Cache reasoning traces across sessions",
                 "Detect drift via rolling perplexity bands"]
    assert tm.validity_gate(rep, items_rep) < tm.validity_gate(diverse_text, items_div)


def test_validity_gate_in_unit_range():
    g = tm.validity_gate("a normal varied sentence about oceans and code",
                         ["a", "b", "c"])
    assert 0.0 <= g <= 1.0
