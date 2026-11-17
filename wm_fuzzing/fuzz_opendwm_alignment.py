"""Fuzz OpenDWM's overlapping CTSD video windows with numbered CPU frames."""

import argparse
import ast
from collections import Counter
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from fuzz_checks import assert_frame_order


SNAPSHOT = Path(__file__).with_name("upstream") / "opendwm"


def load_function(source, name, bindings):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    member = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = dict(bindings)
    exec(compile(ast.Module(body=[member], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


def run_tagged_rollout(rollout, ids, chunk_size, overlap):
    def generate(latent_shape, window, output_type, condition, reference_frame_count):
        tags = window["pts"]
        assert tags.shape == latent_shape
        if condition is not None:
            torch.testing.assert_close(condition, tags[:, :reference_frame_count])
        # The real decoder returns its image batch flattened in sample/time/view order.
        return {"latents": tags, "images": tags.flatten(0, 2)}

    pipeline = SimpleNamespace(
        common_config={},
        inference_config={"sequence_length_per_iteration": chunk_size, "reference_frame_count": overlap},
        inference_pipeline=generate,
        get_latent_sequence_length=int,
    )
    shape = (ids.shape[0], chunk_size, *ids.shape[2:])
    return rollout(pipeline, shape, {"pts": ids}, "pt")["images"]


def run_case(rollout, chunk_size, chunks, overlap, batch_size, views=1):
    stride = chunk_size - overlap
    frames = chunk_size + (chunks - 1) * stride
    ids = torch.arange(frames * views).reshape(1, frames, views)
    ids = ids + 1000 * torch.arange(batch_size).reshape(batch_size, 1, 1)
    ids = ids[..., None, None, None]
    actual = run_tagged_rollout(rollout, ids, chunk_size, overlap).flatten().tolist()
    return ids.flatten().tolist(), actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, help="Use a local OpenDWM checkout instead of the snapshot")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=64)
    args = parser.parse_args()
    if args.cases < 1 or args.batch_size < 1:
        parser.error("cases and batch size must be positive")

    if args.checkout:
        clip_source = args.checkout / "src/dwm/functional.py"
        rollout_source = args.checkout / "src/dwm/pipelines/ctsd.py"
    else:
        clip_source = SNAPSHOT / "take_sequence_clip.py"
        rollout_source = SNAPSHOT / "ctsd_autoregressive.py"
    clip = load_function(clip_source, "take_sequence_clip", {"torch": torch})
    dwm = SimpleNamespace(functional=SimpleNamespace(take_sequence_clip=clip))
    rollout = load_function(rollout_source, "autoregressive_inference_pipeline", {"torch": torch, "dwm": dwm})

    rng = random.Random(args.seed)
    # Match DreamDojo's tested plan. Its action chunk size excludes the reference frame.
    plans = [(13, 2), (13, 3), (13, 4)][:args.cases]
    plans.extend((rng.choice((5, 9, 13, 17)), rng.randint(2, 6)) for _ in range(args.cases - len(plans)))
    cases = [(length, chunks, 1, 1) for length, chunks in plans]
    cases.append((19, 2, 3, 6))  # 35-frame multi-view shape used in OpenDWM's CTSD config.

    failures = 0
    for case, (length, chunks, overlap, views) in enumerate(cases):
        expected, actual = run_case(rollout, length, chunks, overlap, args.batch_size, views)
        try:
            assert_frame_order(expected, actual, f"seed={args.seed}, case={case}, batch_size={args.batch_size}")
        except AssertionError as error:
            if failures == 0:
                print(error)
                want, got = Counter(expected), Counter(actual)
                print(f"Expected {len(expected)} IDs; received {len(actual)} IDs")
                print(f"Missing IDs: {sorted((want - got).elements())}")
                print(f"Extra IDs: {sorted((got - want).elements())}")
            failures += 1

    print(f"OpenDWM batch={args.batch_size}: {len(cases) - failures}/{len(cases)} passed")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
