"""Run Vista's overlapping generation and decode paths with numbered CPU frames."""

import argparse
import ast
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch

from wm_fuzzing.experiments.checks import assert_frame_order
from wm_fuzzing.sources import UPSTREAM


SCALE = 1024


class VideoDecoder:
    """Stand-in type for the temporal decoder's timesteps argument."""


def default(value, fallback):
    return fallback if value is None else value


def load_functions(source, names, scope, class_name=None):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    statements = tree.body
    if class_name:
        statements = next(
            node.body for node in statements if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    functions = [node for node in statements if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)


def identity_decode(latents, *, timesteps):
    assert timesteps == len(latents)
    return latents.clone()


def test_model(decode_first_stage, decode_block):
    model = SimpleNamespace(
        conditioner=SimpleNamespace(embedders=[]),
        decode_first_stage=None,
        denoiser=None,
        disable_first_stage_autocast=True,
        ema_scope=lambda _: nullcontext(),
        en_and_decode_n_samples_a_time=decode_block,
        encode_first_stage=lambda images: images,
        first_stage_model=SimpleNamespace(decoder=VideoDecoder(), decode=identity_decode),
        model=None,
        scale_factor=1,
    )
    model.decode_first_stage = lambda latents: decode_first_stage(model, latents)
    return model


def run_case(do_sample, model, frames, rounds, initial_frames):
    calls = 0

    def sample(_, noise, *, cond, uc, cond_frame, cond_mask):
        nonlocal calls
        start = calls * (frames - 3)
        mask_length = 3 if calls else initial_frames
        result = torch.arange(start, start + frames, dtype=noise.dtype).reshape_as(noise) / SCALE
        assert torch.equal(cond_mask.nonzero().flatten(), torch.arange(mask_length))
        assert torch.equal(cond_frame[:mask_length], result[:mask_length])
        calls += 1
        return result

    images = torch.arange(frames, dtype=torch.float32).reshape(-1, 1, 1, 1) / SCALE
    video, latents, _ = do_sample(images, model, sample, {}, rounds, frames,
                                  initial_cond_indices=list(range(initial_frames)), device="cpu")
    expected = range(frames + (rounds - 1) * (frames - 3))
    context = f"frames={frames}, rounds={rounds}, initial_frames={initial_frames}"
    assert calls == rounds
    assert_frame_order(expected, (latents.flatten() * SCALE).tolist(), f"generation stitch, {context}")
    assert_frame_order(expected, ((video.flatten() * 2 - 1) * SCALE).tolist(), f"decode stitch, {context}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", type=Path, default=UPSTREAM / "Vista")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    decode = {"torch": torch, "default": default, "VideoDecoder": VideoDecoder}
    load_functions(args.repo / "vwm/models/diffusion.py", {"decode_first_stage"}, decode, "DiffusionEngine")
    sampling = {
        "torch": torch,
        "default": default,
        "Optional": Optional,
        "List": List,
        "autocast": lambda _: nullcontext(),
        "get_condition": lambda *args: (None, None),
        "load_model": lambda _: None,
        "unload_model": lambda _: None,
        "tqdm": lambda **kwargs: SimpleNamespace(update=lambda _: None),
    }
    load_functions(args.repo / "sample_utils.py", {"do_sample", "fill_latent"}, sampling)

    rng = random.Random(args.seed)
    cases = [(25, 2, 1), (25, 6, 1), (25, 6, 3)][:args.cases]
    cases.extend(
        (rng.randint(14, 36), rng.randint(2, 8), rng.choice((1, 3)))
        for _ in range(args.cases - len(cases))
    )
    model = test_model(decode["decode_first_stage"], decode_block=14)
    for case in cases:
        run_case(sampling["do_sample"], model, *case)
    print(f"Vista generation and decode: {len(cases)}/{len(cases)} numbered multi-chunk cases passed")


if __name__ == "__main__":
    main()
