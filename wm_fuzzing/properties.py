"""The three model-independent comparisons used by the CPU reproducers."""

from collections.abc import Callable, Sequence
from typing import TypeVar

Sample = TypeVar("Sample")
Result = TypeVar("Result")
Instance = TypeVar("Instance")
Snapshot = TypeVar("Snapshot")


def assert_batch_isolation(
    run: Callable[[list[Sample]], Sequence[Result]],
    batch: Sequence[Sample],
    replacement: Sample,
    *,
    held: int = 0,
    changed: int = 1,
) -> None:
    """Check that one sample's replacement leaves another sample's output identical."""
    if not (0 <= held < len(batch) and 0 <= changed < len(batch)) or held == changed:
        raise ValueError("held and changed must be different valid batch indices")

    original = run(list(batch))
    updated = list(batch)
    updated[changed] = replacement
    rerun = run(updated)
    if len(original) != len(batch) or len(rerun) != len(batch):
        raise AssertionError("Expected one output per sample")
    if original[held] != rerun[held]:
        raise AssertionError(
            f"Changing sample {changed} changed untouched sample {held}\n"
            f"Before: {original[held]}\nAfter: {rerun[held]}"
        )


def check_reset_isolation(
    make: Callable[[], Instance],
    reset: Callable[[Instance], object],
    warm: Callable[[Instance], object],
    probe: Callable[[Instance], Result],
) -> tuple[Result, Result]:
    """Return matching probes of a fresh instance and a warmed, then reset, instance.

    The warm callback initializes the used instance and gives it a prior history.
    Both instances receive the same final reset settings and probe inputs.
    """
    fresh, used = make(), make()
    reset(fresh)
    warm(used)
    reset(used)
    return probe(fresh), probe(used)


def check_restore_replay(
    instance: Instance,
    save: Callable[[Instance], Snapshot],
    restore: Callable[[Instance, Snapshot], object],
    run: Callable[[Instance], Result],
) -> tuple[Result, Result]:
    """Return traces from the original continuation and the same inputs after restore."""
    snapshot = save(instance)
    expected = run(instance)
    restore(instance, snapshot)
    return expected, run(instance)
