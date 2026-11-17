"""Additional checks for frame order and finite action normalization."""

import math
import random
from collections.abc import Callable, Sequence


def assert_frame_order(expected: Sequence[int], actual: Sequence[int], context: str) -> None:
    """Compare frame identities so an equal-length repeat and drop still fails."""
    expected_ids = list(expected)
    actual_ids = list(actual)
    if actual_ids != expected_ids:
        raise AssertionError(
            f"Frame order mismatch ({context})\n"
            f"Expected {expected_ids}\n"
            f"Received {actual_ids}"
        )


def check_frame_stitching(
    stitch: Callable[[list[list[int]]], Sequence[int]],
    *,
    seed: int = 42,
    cases: int = 32,
) -> None:
    """Check that each frame appears once, in order, across one-frame overlaps."""
    if cases < 1:
        raise ValueError("cases must be positive")

    rng = random.Random(seed)
    for case in range(cases):
        lengths = [13, 13]
        if case > 0:
            lengths = [rng.randint(2, 16) for _ in range(rng.randint(2, 5))]
        chunks = []
        start = 0
        for length in lengths:
            chunks.append(list(range(start, start + length)))
            start += length - 1

        expected = list(range(start + 1))
        assert_frame_order(expected, stitch(chunks), f"seed={seed}, case={case}, lengths={lengths}")


def check_action_normalization(
    normalize: Callable[[list[list[float]]], Sequence[Sequence[float]]],
    *,
    seed: int = 42,
    cases: int = 32,
) -> None:
    """Check that normalization preserves shape and returns finite actions."""
    if cases < 1:
        raise ValueError("cases must be positive")

    rng = random.Random(seed)
    for case in range(cases):
        if case == 0:
            actions = [[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]]
            rows, channels = 3, 2
            fixed_channel = 1
        else:
            rows, channels = rng.randint(2, 8), rng.randint(1, 6)
            fixed_channel = rng.randrange(channels)
            fixed_value = rng.uniform(-8, 8)
            actions = [
                [
                    fixed_value if channel == fixed_channel else rng.uniform(-8, 8)
                    for channel in range(channels)
                ]
                for _ in range(rows)
            ]

        actual = normalize(actions)
        context = f"seed={seed}, case={case}, fixed_channel={fixed_channel}"
        if len(actual) != rows or any(len(row) != channels for row in actual):
            raise AssertionError(f"Expected action shape ({rows}, {channels}) ({context})")

        for step, row in enumerate(actual):
            for channel, value in enumerate(row):
                if not math.isfinite(value):
                    raise AssertionError(
                        f"Non-finite action at step {step}, channel {channel}: {value} ({context})"
                    )
