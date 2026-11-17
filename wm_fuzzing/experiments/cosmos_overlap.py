"""Check the Cosmos-Predict2.5 sliding-window stitch with numbered CPU frames."""

import argparse
import ast
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from wm_fuzzing.experiments.checks import assert_frame_order
from wm_fuzzing.sources import UPSTREAM


DEFAULT_SOURCE = UPSTREAM / "cosmos_predict2_5" / "video2world.py"


def load_rollout(source):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    inference = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Video2WorldInference"
    )
    method = next(
        node for node in inference.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_autoregressive_from_batch"
    )
    scope = {"torch": torch, "log": logging.getLogger(__name__), "_DEFAULT_NEGATIVE_PROMPT": ""}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    return scope[method.name]


def run_case(rollout, frames, chunk_size, overlap):
    rollout_seed = 100

    def latent_frames(pixel_frames):
        return 1 + (pixel_frames - 1) // 4

    def pixel_frames(latent_count):
        return 1 + 4 * (latent_count - 1)

    def generate(input_path, num_video_frames, num_latent_conditional_frames, seed, **kwargs):
        start = (seed - rollout_seed) * (chunk_size - overlap)
        assert num_video_frames == chunk_size
        assert num_latent_conditional_frames == latent_frames(overlap)
        prefix = input_path[0, 0, :overlap, 0, 0].tolist()
        assert_frame_order(range(start, start + overlap), prefix, f"conditioning prefix at {start}")
        ids = torch.arange(start, start + chunk_size, dtype=torch.float64)
        # Retain frame IDs through the upstream float-to-uint8 conversion.
        pixels = (ids + 0.25) * (2.0 / 255.0) - 1.0
        return pixels.reshape(1, 1, -1, 1, 1).expand(1, 3, -1, 1, 1)

    tokenizer = SimpleNamespace(get_latent_num_frames=latent_frames, get_pixel_num_frames=pixel_frames)
    model = SimpleNamespace(tokenizer=tokenizer, config=SimpleNamespace(state_t=latent_frames(chunk_size)))
    inference = SimpleNamespace(model=model, generate_vid2world=generate)
    initial = torch.arange(overlap, dtype=torch.uint8).reshape(1, 1, -1, 1, 1).expand(1, 3, -1, 1, 1)
    output = rollout(
        inference, prompt="", input_path=initial, num_output_frames=frames,
        chunk_size=chunk_size, chunk_overlap=overlap,
        num_latent_conditional_frames=latent_frames(overlap), resolution="1,1", seed=rollout_seed,
    )
    return ((output[0, 0, :, 0, 0] + 1.0) * 127.5).to(torch.int64).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=64)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    rollout = load_rollout(args.source)
    rng = random.Random(args.seed)
    cases = [(25, 13, 1), (49, 13, 1), (153, 77, 1), (173, 77, 5)][:args.cases]
    while len(cases) < args.cases:
        overlap = rng.choice((1, 5))
        size = rng.choice((13, 17, 21, 77))
        frames = rng.randint(size + 1, min(178, size + 5 * (size - overlap)))
        cases.append((frames, size, overlap))

    with torch.inference_mode():
        for case, (frames, size, overlap) in enumerate(cases):
            result = run_case(rollout, frames, size, overlap)
            assert_frame_order(
                range(frames), result,
                f"seed={args.seed}, case={case}, frames={frames}, chunk_size={size}, overlap={overlap}",
            )
    print(f"Cosmos-Predict2.5: {len(cases)}/{len(cases)} overlapping-window cases passed")


if __name__ == "__main__":
    main()
