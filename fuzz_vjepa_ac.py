"""Fuzz V-JEPA 2-AC on CPU. Usage: python fuzz_vjepa_ac.py PATH_TO_VJEPA2."""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import torch


def check_rotation(rotate, rng):
    batch = rng.randint(1, 3)
    heads = rng.randint(1, 4)
    tokens = rng.randint(1, 8)
    width = 2 * rng.randint(1, 12)
    x = torch.randn(batch, heads, tokens, width, dtype=torch.float64)
    positions = torch.randint(0, 1024, (tokens,)).to(x.dtype)
    y = rotate(x, positions)
    # Each adjacent coordinate pair should undergo a length-preserving rotation.
    squared_lengths = lambda value: value.unflatten(-1, (-1, 2)).square().sum(-1)
    try:
        torch.testing.assert_close(squared_lengths(y), squared_lengths(x))
    except AssertionError:
        error = (squared_lengths(y) - squared_lengths(x)).abs().max().item()
        raise AssertionError(
            f"shape={tuple(x.shape)}, positions={positions.tolist()}, "
            f"max squared-length error={error:.6g}"
        ) from None


def check_predictor(predictor, rng, causal):
    batch = rng.randint(1, 2)
    steps = rng.randint(2, 5)
    grid = rng.randint(1, 3)
    patches = grid**2
    use_extrinsics = rng.choice((True, False))
    model = predictor(
        img_size=grid * 16,
        num_frames=steps * 2,
        tubelet_size=2,
        embed_dim=16,
        predictor_embed_dim=48,
        depth=rng.randint(1, 2),
        num_heads=2,
        use_extrinsics=use_extrinsics,
        is_frame_causal=causal,
    ).eval()
    x = torch.randn(batch, steps * patches, 16)
    actions = torch.randn(batch, steps, 7)
    states = torch.randn_like(actions)
    extrinsics = torch.randn(batch, steps, 6) if use_extrinsics else None
    y = model(x, actions, states, extrinsics)
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise AssertionError(f"invalid output for steps={steps}, grid={grid}")

    if causal:
        cut = rng.randint(1, steps - 1)
        changed = []
        for value, prefix in (
            (x, cut * patches), (actions, cut), (states, cut), (extrinsics, cut)
        ):
            if value is None:
                changed.append(None)
                continue
            mutated = value.clone()
            mutated[:, prefix:] = 3 * torch.randn_like(mutated[:, prefix:])
            changed.append(mutated)
        new_y = model(*changed)
        torch.testing.assert_close(new_y[:, : cut * patches], y[:, : cut * patches])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="Local checkout of facebookresearch/vjepa2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=100, help="Cases per property")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    if not (args.repo / "src/models/ac_predictor.py").is_file():
        parser.error("repo must contain src/models/ac_predictor.py")

    sys.path.insert(0, str(args.repo.resolve()))
    from src.models.ac_predictor import VisionTransformerPredictorAC
    from src.models.utils.modules import rotate_queries_or_keys

    torch.set_num_threads(1)
    checks = {
        "rotation preserves pair lengths": lambda rng: check_rotation(rotate_queries_or_keys, rng),
        "causal prefix ignores future inputs": lambda rng: check_predictor(
            VisionTransformerPredictorAC, rng, causal=True
        ),
        "noncausal configuration runs": lambda rng: check_predictor(
            VisionTransformerPredictorAC, rng, causal=False
        ),
    }
    failures = Counter()
    with torch.inference_mode():
        for name, check in checks.items():
            for i in range(args.cases):
                seed = args.seed + i
                torch.manual_seed(seed)
                try:
                    check(random.Random(seed))
                except Exception as error:
                    if failures[name] == 0:
                        print(f"{name}: seed={seed}, {type(error).__name__}: {error}")
                    failures[name] += 1
            print(f"{name}: {args.cases - failures[name]}/{args.cases} passed")
    return int(any(failures.values()))


if __name__ == "__main__":
    sys.exit(main())
