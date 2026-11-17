"""Apply batch isolation to live DIAMOND and V-JEPA 2-AC predictor code on CPU."""

import argparse
import ast
import math
import random
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from fuzz_action_routing import SOURCE, load_nodes, load_vjepa, make_vjepa
from fuzz_checks import assert_batch_isolation


def make_diamond_reward_model():
    # DIAMOND 5bcd159, src/models/{blocks,rew_end_model}.py and src/utils.py.
    path = SOURCE / "diamond"
    scope = {"torch": torch, "nn": nn, "F": nn.functional, "partial": partial, "math": math, "dataclass": dataclass}
    load_nodes(path / "blocks.py", scope, lambda node: isinstance(node, (ast.ClassDef, ast.Assign)))
    load_nodes(path / "init_lstm.py", scope, lambda node: isinstance(node, ast.FunctionDef))
    load_nodes(path / "rew_end_model.py", scope, lambda node: isinstance(node, ast.ClassDef))
    config = scope["RewEndModelConfig"](
        lstm_dim=8, img_channels=3, img_size=4, cond_channels=8,
        depths=[1, 1], channels=[8, 8], attn_depths=[False, True], num_actions=6,
    )
    return scope["RewEndModel"](config).eval()


def diamond_case(model, rng, size):
    steps = rng.randint(2, 7)
    batch = [
        (torch.randn(steps, 3, 4, 4), torch.randint(6, (steps,)), torch.randn(steps, 3, 4, 4), torch.randn(2, 8))
        for _ in range(size)
    ]

    def run(samples):
        observed, actions, next_observed, memory = (torch.stack(field) for field in zip(*samples))
        hx, cx = (part.unsqueeze(0).contiguous() for part in memory.unbind(dim=1))
        reward, end, (hx, cx) = model.predict_rew_end(observed, actions, next_observed, (hx, cx))
        return torch.cat((reward.flatten(1), end.flatten(1), hx[0], cx[0]), dim=1).tolist()

    held, changed = rng.sample(range(size), 2)
    obs, actions, next_obs, memory = batch[changed]
    replacement = (obs + 2, (actions + 1) % 6, next_obs - 1, memory * -2)
    assert_batch_isolation(run, batch, replacement, held=held, changed=changed)


def vjepa_case(model_type, rng, size):
    steps, grid = rng.randint(2, 7), rng.randint(1, 2)
    camera = rng.choice((True, False))
    model = make_vjepa(model_type, steps, grid, extrinsics=camera)
    shapes = [(steps * grid**2, 16), (steps, 7), (steps, 7)]
    if camera:
        shapes.append((steps, 6))
    batch = [tuple(torch.randn(shape) for shape in shapes) for _ in range(size)]

    def run(samples):
        return model(*(torch.stack(field) for field in zip(*samples))).flatten(1).tolist()

    held, changed = rng.sample(range(size), 2)
    replacement = list(batch[changed])
    field = rng.randrange(len(replacement))
    replacement[field] = replacement[field] * -3 + 2
    assert_batch_isolation(run, batch, tuple(replacement), held=held, changed=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=32)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checks = [
        ("DIAMOND reward/end predictor", make_diamond_reward_model(), diamond_case),
        ("V-JEPA 2-AC causal predictor", load_vjepa(), vjepa_case),
    ]
    failed = False
    with torch.inference_mode():
        for name, model, check in checks:
            failures = 0
            for case in range(args.cases):
                seed = args.seed + case
                torch.manual_seed(seed)
                try:
                    check(model, random.Random(seed), 2 + case % 3)
                except AssertionError as error:
                    if failures == 0:
                        print(f"{name}, seed={seed}: {error}")
                    failures += 1
            print(f"{name}: {args.cases - failures}/{args.cases} passed")
            failed |= failures > 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
