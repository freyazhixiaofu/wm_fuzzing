"""Run two upstream latent rollouts with tagged observations and toy predictors."""

import argparse
import ast
import contextlib
import io
import random
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
from wm_fuzzing.adapters.latent_rollouts import (
    DINO_SOURCE,
    LE_SOURCE,
    run_dino_rollout,
    run_le_rollout,
)


def numbered_rollout_inputs(rng, case):
    if case == 0:
        initial = torch.tensor([[[10.0]]])
        actions = torch.tensor([[[1.0, 2.0, 4.0]]])
        observed, lookback = 1, 1
    else:
        batches, plans = rng.randint(1, 3), rng.randint(1, 4)
        observed, lookback = rng.randint(1, 6), rng.randint(1, 6)
        forecast = rng.randint(1, 9)
        initial = torch.arange(batches, dtype=torch.float32).view(batches, 1, 1) * 100 + 10
        actions = torch.zeros(batches, plans, observed - 1 + forecast)
        for batch in range(batches):
            for t in range(observed - 1):
                actions[batch, :, t] = rng.randrange(-4, 5)
            for plan in range(plans):
                future = actions[batch, plan, observed - 1 :]
                if case % 2:
                    future[rng.randrange(forecast)] = 2 + batch + plan
                else:
                    future[:] = torch.tensor([rng.randrange(-8, 9) for _ in range(forecast)])

    # By construction, each action moves its state tag by exactly the action's value.
    expected = torch.cat((initial.expand_as(actions[..., :1]), initial + actions.cumsum(2)), 2)
    frames = expected[:, :, :observed, None, None, None].clone()
    return frames, actions.unsqueeze(-1), expected, lookback


def load_definitions(path, names, namespace):
    """Load selected upstream definitions without unrelated data or training dependencies."""
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source.body = [node for node in source.body if getattr(node, "name", None) in names]
    exec(compile(source, str(path), "exec"), namespace)
    return [namespace[name] for name in names]


def show_normalization(dino_source, le_source):
    import numpy as np
    from typing import Callable, Optional

    actions = torch.tensor([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    fixture = {
        "states.pth": torch.tensor([[[0.0], [1.0], [2.0]]]),
        "actions.pth": actions.unsqueeze(0),
        "door_locations.pth": torch.zeros((1, 3)),
        "wall_locations.pth": torch.zeros((1, 3)),
    }
    tensor_ops = SimpleNamespace(
        load=lambda path: fixture[path.name].clone(), zeros=torch.zeros, ones=torch.ones
    )
    wall_dataset, = load_definitions(
        dino_source.parent.parent / "datasets" / "wall_dset.py",
        ["WallDataset"],
        {
            "torch": tensor_ops, "Path": Path, "TrajDataset": object,
            "Optional": Optional, "Callable": Callable,
        },
    )
    with contextlib.redirect_stdout(io.StringIO()):
        default = wall_dataset().actions[0, :, 1]
        normalized = wall_dataset(normalize_action=True).actions[0, :, 1]

    # The third-party dataset wrapper only needs to pass through the upstream normalizer.
    transforms = SimpleNamespace(WrapTorchTransform=lambda normalizer, **_: normalizer)
    get_column_normalizer, _ = load_definitions(
        le_source.with_name("utils.py"),
        ["get_column_normalizer", "ZScoreNormalizer"],
        {"torch": torch, "np": np, "dt": SimpleNamespace(transforms=transforms)},
    )
    dataset = SimpleNamespace(get_col_data=lambda _: actions.numpy())
    standardized = get_column_normalizer(dataset, "action", "action")(actions)

    class ReachedEncoder(Exception):
        pass

    def capture_encoder_input(batch):
        raise ReachedEncoder(batch["action"])

    training_forward, = load_definitions(
        le_source.with_name("train.py"), ["lejepa_forward"], {"torch": torch},
    )
    cfg = SimpleNamespace(
        history_size=1, num_preds=1, loss=SimpleNamespace(sigreg=SimpleNamespace(weight=1))
    )
    model = SimpleNamespace(model=SimpleNamespace(encode=capture_encoder_input))
    try:
        training_forward(model, {"action": standardized.clone()}, "train", cfg)
    except ReachedEncoder as captured:
        encoder_input = captured.args[0]
    else:
        raise RuntimeError("LeWorldModel did not call the encoder")

    print(f"DINO-WM WallDataset default, constant channel: {default.tolist()}")
    print(f"DINO-WM WallDataset with normalize_action=True, constant channel: {normalized.tolist()}")
    print(f"LeWorldModel z-score normalizer, constant channel: {standardized[:, 1].tolist()}")
    print(f"LeWorldModel training encoder input, constant channel: {encoder_input[:, 1].tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-source", type=Path, default=DINO_SOURCE)
    parser.add_argument("--le-source", type=Path, default=LE_SOURCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument(
        "--show-normalization", action="store_true",
        help="Also reproduce the separate constant-action preprocessing behavior",
    )
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    dino = runpy.run_path(str(args.dino_source))["VWorldModel"]
    le = runpy.run_path(str(args.le_source))["JEPA"]
    rng = random.Random(args.seed)
    with torch.inference_mode():
        for case in range(args.cases):
            frames, actions, expected, lookback = numbered_rollout_inputs(rng, case)
            try:
                for concat_dim in (0, 1):
                    actual = run_dino_rollout(dino, frames, actions, lookback, concat_dim)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                actual = run_le_rollout(le, frames, actions, lookback)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            except AssertionError as error:
                raise AssertionError(f"Alignment failure (seed={args.seed}, case={case})") from error

    print(f"DINO-WM: {args.cases * 2}/{args.cases * 2} passed across concat_dim=0 and concat_dim=1")
    print(f"LeWorldModel: {args.cases}/{args.cases} passed")
    if args.show_normalization:
        show_normalization(args.dino_source, args.le_source)


if __name__ == "__main__":
    main()
