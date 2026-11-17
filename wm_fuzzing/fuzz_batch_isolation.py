"""Test whether an independent sample can change another sample's output."""

import argparse
import random
import runpy
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import torch
from einops import rearrange

from fuzz_checks import assert_batch_isolation
from fuzz_latent_alignment import run_dino_rollout, run_le_rollout
from fuzz_opendwm_alignment import load_function, run_tagged_rollout


UPSTREAM = Path(__file__).with_name("upstream")


def load_dreamdojo():
    # NVIDIA/DreamDojo at 02f119b, text2world_model_rectified_flow.py lines 863–869.
    path = UPSTREAM / "DreamDojo" / "lam_forward_excerpt.txt"
    stage = compile(dedent(path.read_text(encoding="utf-8")), str(path), "exec")

    def lam(inputs):
        # The real LAM sees a flattened batch of two-frame pairs.
        tags = inputs["videos"][:, 0, 0, 0, 0]
        return {"z_rep": tags[:, None, None].expand(-1, 1, 32)}

    def run(batch):
        tags = torch.tensor(batch, dtype=torch.float32)
        samples, steps = tags.shape
        data = {
            "lam_video": tags.repeat_interleave(2, dim=1)[..., None, None, None],
            "action": torch.ones(samples, steps, 32),
        }
        model = SimpleNamespace(lam=SimpleNamespace(lam=lam))
        exec(stage, {"torch": torch, "rearrange": rearrange, "self": model, "data_batch": data})
        return data["action"][..., 0].tolist()

    return run


def load_opendwm():
    path = UPSTREAM / "opendwm"
    clip = load_function(path / "take_sequence_clip.py", "take_sequence_clip", {"torch": torch})
    dwm = SimpleNamespace(functional=SimpleNamespace(take_sequence_clip=clip))
    rollout = load_function(path / "ctsd_autoregressive.py", "autoregressive_inference_pipeline", {"torch": torch, "dwm": dwm})

    def run(batch):
        tags = torch.tensor(batch)[..., None, None, None, None]
        chunk_size = (tags.shape[1] + 1) // 2
        result = run_tagged_rollout(rollout, tags, chunk_size, overlap=1)
        # Match the metric consumer's batch-major interpretation of the results.
        return result.reshape(len(batch), -1).tolist()

    return run


def make_latent_adapter(rollout):
    def run(batch):
        actions = torch.tensor(batch, dtype=torch.float32)[:, None, :, None]
        frames = torch.full((len(batch), 1, 1, 1, 1, 1), 10.0)
        return rollout(frames, actions)[:, 0].tolist()

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=32)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    dino = runpy.run_path(str(UPSTREAM / "dino_wm/models/visual_world_model.py"))["VWorldModel"]
    le = runpy.run_path(str(UPSTREAM / "le-wm/jepa.py"))["JEPA"]
    checks = {
        "DreamDojo LAM training conditioning": load_dreamdojo(),
        "OpenDWM CTSD stitched output": load_opendwm(),
        "DINO-WM rollout, token layout": make_latent_adapter(lambda f, a: run_dino_rollout(dino, f, a, 3, 0)),
        "DINO-WM rollout, channel layout": make_latent_adapter(lambda f, a: run_dino_rollout(dino, f, a, 3, 1)),
        "LeWorldModel rollout": make_latent_adapter(lambda f, a: run_le_rollout(le, f, a, 3)),
    }
    rng = random.Random(args.seed)
    cases = [(2, 3)] + [(rng.randint(2, 4), rng.randrange(3, 26, 2)) for _ in range(args.cases - 1)]

    any_failure = False
    with torch.inference_mode():
        for name, run in checks.items():
            failures = 0
            for case, (count, steps) in enumerate(cases):
                batch = [list(range(i * 100 + 1, i * 100 + steps + 1)) for i in range(count)]
                replacement = [value + 1000 for value in batch[1]]
                try:
                    assert_batch_isolation(run, batch, replacement)
                except AssertionError as error:
                    if not failures:
                        print(f"{name}, seed={args.seed}, case={case}: {error}")
                    failures += 1
            print(f"{name}: {len(cases) - failures}/{len(cases)} passed")
            any_failure |= failures > 0
    return int(any_failure)


if __name__ == "__main__":
    raise SystemExit(main())
