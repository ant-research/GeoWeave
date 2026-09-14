"""Conservative detection of exact token-level repetition loops.

The detector is intended for length-capped RL rollouts.  It only reports a
loop when at least ``min_repeats`` adjacent copies of a token block are found
and that repeated run reaches the response tail (allowing a small incomplete
suffix).  This avoids treating ordinary references to an earlier formula as a
recoverable generation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class RepetitionLoop:
    """One high-confidence adjacent token repetition near the response tail."""

    # Keep tokens [0:cutoff_token).  This retains the first copy and removes the
    # second and later copies of the repeated block.
    cutoff_token: int
    period_tokens: int
    repeated_region_start: int
    repeated_region_end: int
    full_repeats: int

    @property
    def removed_from(self) -> int:
        return self.cutoff_token


def detect_tandem_token_repetition(
    token_ids: Sequence[int],
    *,
    min_block_tokens: int = 16,
    max_block_tokens: int = 256,
    min_repeats: int = 3,
    min_prefix_tokens: int = 64,
    tail_tolerance_tokens: int = 32,
) -> Optional[RepetitionLoop]:
    """Find an exact adjacent repetition loop that continues to the tail.

    For a period ``p``, comparing ``tokens[i]`` with ``tokens[i-p]`` converts a
    repeated block into a consecutive equality run.  A run of at least
    ``p * (min_repeats - 1)`` comparisons proves at least ``min_repeats`` copies.

    The earliest safe cutoff is preferred: it keeps the first block and removes
    the second and later copies.  Ties prefer the shorter fundamental period.
    """
    ids = [int(token_id) for token_id in token_ids]
    n = len(ids)
    if min_block_tokens <= 0:
        raise ValueError("min_block_tokens must be positive")
    if max_block_tokens < min_block_tokens:
        raise ValueError("max_block_tokens must be >= min_block_tokens")
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")
    if min_prefix_tokens < 0 or tail_tolerance_tokens < 0:
        raise ValueError("prefix/tail thresholds must be non-negative")

    max_period = min(int(max_block_tokens), n // min_repeats)
    if max_period < min_block_tokens:
        return None

    candidates: list[RepetitionLoop] = []
    for period in range(int(min_block_tokens), max_period + 1):
        run_start = period
        run_length = 0

        def consider(compare_end: int) -> None:
            # Equality comparisons span [run_start, compare_end).  The first
            # copy starts one period earlier.
            nonlocal run_length
            required = period * (int(min_repeats) - 1)
            if run_length < required:
                return
            repeated_start = run_start - period
            repeated_end = compare_end
            cutoff = run_start
            if cutoff < int(min_prefix_tokens):
                return
            if repeated_end < n - int(tail_tolerance_tokens):
                return
            full_repeats = (repeated_end - repeated_start) // period
            if full_repeats < int(min_repeats):
                return
            candidates.append(
                RepetitionLoop(
                    cutoff_token=cutoff,
                    period_tokens=period,
                    repeated_region_start=repeated_start,
                    repeated_region_end=repeated_end,
                    full_repeats=full_repeats,
                )
            )

        for index in range(period, n):
            if ids[index] == ids[index - period]:
                if run_length == 0:
                    run_start = index
                run_length += 1
            else:
                consider(index)
                run_length = 0
        consider(n)

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.cutoff_token, item.period_tokens))


__all__ = ["RepetitionLoop", "detect_tandem_token_repetition"]
