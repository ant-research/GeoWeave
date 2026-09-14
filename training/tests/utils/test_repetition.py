from unirl.utils.repetition import detect_tandem_token_repetition


def test_detects_three_copy_tail_loop_and_keeps_first_copy():
    prefix = list(range(80))
    block = list(range(100, 124))
    loop = detect_tandem_token_repetition(prefix + block * 3)

    assert loop is not None
    assert loop.cutoff_token == len(prefix) + len(block)
    assert loop.period_tokens == len(block)
    assert loop.full_repeats == 3


def test_detects_loop_with_incomplete_tail_copy():
    prefix = list(range(80))
    block = list(range(100, 132))
    loop = detect_tandem_token_repetition(
        prefix + block * 3 + block[:10], tail_tolerance_tokens=16
    )

    assert loop is not None
    assert loop.cutoff_token == len(prefix) + len(block)


def test_ignores_repetition_that_does_not_reach_tail():
    prefix = list(range(80))
    block = list(range(100, 124))
    suffix = list(range(500, 600))

    assert detect_tandem_token_repetition(prefix + block * 3 + suffix) is None


def test_ignores_short_or_two_copy_repetition():
    prefix = list(range(80))
    assert detect_tandem_token_repetition(prefix + [1, 2, 3, 4] * 8) is None
    assert detect_tandem_token_repetition(prefix + list(range(100, 120)) * 2) is None
