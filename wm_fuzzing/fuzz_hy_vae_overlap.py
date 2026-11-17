"""Check numbered frames through HY-WorldPlay's overlapping temporal VAE tiles."""

import argparse
import ast
import json
import random
from pathlib import Path

import torch

from fuzz_checks import assert_frame_order


def load_decode_methods(source):
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ParallelTiledVAE")
    names = {"decode", "tiled_decode", "blend_t"}
    definitions = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch, "get_sp_world_size": lambda: 1}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), scope)
    return {name: scope[name] for name in names}


def numbered_decode(self, tile):
    # FastHunyuan keeps the tile's first anchor frame, then expands each later latent into four frames.
    first = int(tile[0, 0, 0, 0, 0]) * self.temporal_compression_ratio
    last = int(tile[0, 0, -1, 0, 0]) * self.temporal_compression_ratio
    self.decoded_windows.append((first, last))
    return torch.arange(first, last + 1, dtype=torch.float64).view(1, 1, -1, 1, 1)


def check_overlap(source, recipe, seed, cases):
    options = json.loads(recipe.read_text(encoding="utf-8"))["vae_config"]
    methods = load_decode_methods(source)
    methods["_decode"] = numbered_decode
    numbered_vae = type("NumberedVAE", (), methods)()
    numbered_vae.__dict__.update(options)
    numbered_vae.temporal_compression_ratio = 4
    numbered_vae.spatial_compression_ratio = 8

    rng = random.Random(seed)
    lengths = [5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 19]
    lengths = lengths[:cases] + [rng.randint(5, 96) for _ in range(max(cases - len(lengths), 0))]
    failures = 0
    for case, latent_count in enumerate(lengths):
        z = torch.arange(latent_count).view(1, 1, -1, 1, 1)
        numbered_vae.decoded_windows = []
        actual = numbered_vae.decode(z).flatten().tolist()
        intervals = numbered_vae.decoded_windows
        if len(intervals) < 2 or not all(right[0] <= left[1] for left, right in zip(intervals, intervals[1:])):
            raise RuntimeError("Fixture did not reach the overlapping temporal decode path")
        expected = range((latent_count - 1) * numbered_vae.temporal_compression_ratio + 1)
        try:
            assert_frame_order(expected, actual, f"seed={seed}, case={case}, {latent_count=}")
        except AssertionError as error:
            if failures == 0:
                print(error)
            failures += 1
    print(f"HY-WorldPlay temporal VAE: {len(lengths) - failures}/{len(lengths)} overlapping cases passed")
    return int(failures > 0)


def main():
    snapshot = Path(__file__).resolve().parent / "upstream" / "hy_worldplay"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=snapshot / "temporal_vae_common.py")
    parser.add_argument("--recipe", type=Path, default=snapshot / "fasthunyuan_t2v.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=64)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    return check_overlap(args.source, args.recipe, args.seed, args.cases)


if __name__ == "__main__":
    raise SystemExit(main())
