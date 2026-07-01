from bench import report


def test_pearson_perfect_positive():
    assert abs(report.pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9


def test_pearson_none_when_degenerate():
    assert report.pearson([1, 1, 1], [2, 3, 4]) is None
    assert report.pearson([1.0], [2.0]) is None


def test_correlation_card_mentions_axes():
    class R:
        def __init__(self, s): self.score = s
    scored = {"manifest": {"models": ["claude-opus-4-8", "claude-opus-4-6", "claude-sonnet-5"]},
              "scores": {"creativity": {"claude-opus-4-8": R(60), "claude-opus-4-6": R(57), "claude-sonnet-5": R(52)},
                         "audience": {"claude-opus-4-8": R(94), "claude-opus-4-6": R(96), "claude-sonnet-5": R(96)}}}
    html = report.correlation_card(scored)
    assert "창의" in html and "청중" in html
