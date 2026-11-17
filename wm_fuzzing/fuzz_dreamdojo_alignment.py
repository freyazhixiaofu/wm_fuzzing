"""Run DreamDojo's evaluation chunk loop on CPU with numbered synthetic frames."""

import argparse
import ast
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fuzz_checks import assert_frame_order


def load_eval_loop(source: Path):
    """Extract the upstream loop and stitch, leaving checkpoint setup and file I/O out."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    inference = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inference")
    samples = next(
        node for node in inference.body
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "eval_indices"
    )
    start = next(i for i, stmt in enumerate(samples.body) if ast.unparse(stmt) == "first_round = True")
    end = next(
        i for i, stmt in enumerate(samples.body)
        if ast.unparse(stmt).startswith("chunk_video = np.concatenate(")
    )
    return compile(ast.Module(body=samples.body[start : end + 1], type_ignores=[]), str(source), "exec")


def numbered_video(ids):
    # Frame values survive the upstream [-1, 1] to uint8 conversion.
    encoded = (ids.to(torch.float64) + 0.25) * (2.0 / 255.0) - 1.0
    return encoded.reshape(1, 1, -1, 1, 1).expand(1, 3, -1, 1, 1)


def run_eval_case(loop, chunk_size, chunk_count):
    def generate(input_path, action, lam_video, num_video_frames, num_latent_conditional_frames, **kwargs):
        ids = action[:, 0].to(torch.int64)
        conditioning_id = int(input_path[0, 0, 0, 0, 0])
        assert conditioning_id == int(ids[0]) - 1
        assert num_video_frames == len(ids) + 1
        assert num_latent_conditional_frames == 1
        assert lam_video.tolist() == list(range(2 * conditioning_id, 2 * int(ids[-1])))
        return numbered_video(torch.cat((torch.tensor([conditioning_id]), ids)))

    action_count = chunk_size * chunk_count
    actions = np.zeros((action_count, 7), dtype=np.float32)
    actions[:, 0] = np.arange(1, action_count + 1)
    transforms = SimpleNamespace(
        functional=SimpleNamespace(to_tensor=lambda image: torch.from_numpy(image).permute(2, 0, 1).float() / 255)
    )
    scope = {
        "np": np,
        "torch": torch,
        "torchvision": SimpleNamespace(transforms=transforms),
        "video2world_cli": SimpleNamespace(generate_vid2world=generate),
        "inference_args": SimpleNamespace(
            start_frame_idx=0, chunk_size=chunk_size, guidance=7,
            num_latent_conditional_frames=1, negative_prompt="", single_chunk=False,
        ),
        "img_array": torch.zeros(1, 3, 1, 1, dtype=torch.uint8),
        "actions": actions,
        "lam_video": torch.arange(2 * action_count),
        "frames": [],
        "chunk_video": [],
    }
    exec(loop, scope)
    return scope["chunk_video"][:, 0, 0, 0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="Path to a local DreamDojo source checkout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=64, help="Number of multi-chunk cases")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    source = args.repo / "cosmos_predict2/action_conditioned.py"
    if not source.is_file():
        parser.error(f"source file not found at {source}")

    loop = load_eval_loop(source)
    rng = random.Random(args.seed)
    chunk_sizes = (4, 8, 12, 16)
    # Include the minimal repro, the default 37 frames, and eval.sh's 49 frames.
    plans = [(12, 2), (12, 3), (12, 4)][:args.cases]
    plans.extend((rng.choice(chunk_sizes), rng.randint(2, 6)) for _ in range(args.cases - len(plans)))
    for size in chunk_sizes:
        actual = run_eval_case(loop, size, 1)
        assert_frame_order(range(size + 1), actual, f"single chunk, size={size}")
    print(f"Single-chunk controls: {len(chunk_sizes)}/{len(chunk_sizes)} passed")

    failures = 0
    same_length = 0
    for case, (size, count) in enumerate(plans):
        actual = run_eval_case(loop, size, count)
        expected = range(size * count + 1)
        try:
            assert_frame_order(expected, actual, f"seed={args.seed}, case={case}, chunk_size={size}, chunks={count}")
        except AssertionError as error:
            if failures == 0:
                print(error)
            failures += 1
            same_length += len(actual) == len(expected)
    print(f"Multi-chunk cases: {len(plans) - failures}/{len(plans)} passed")
    print(f"Equal-length failures: {same_length}/{failures}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
