"""Run CPU checks against the action preprocessing taken from five upstream repos."""

import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Callable, Optional

from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F

from fuzz_latent_alignment import load_definitions


UPSTREAM = Path(__file__).resolve().parent / "upstream"


def load_dino_normalization():
    fixture = {}
    tensor_ops = SimpleNamespace(
        load=lambda path: fixture[path.name].clone(), zeros=torch.zeros, ones=torch.ones
    )
    wall_dataset, = load_definitions(
        UPSTREAM / "dino_wm/datasets/wall_dset.py", ["WallDataset"],
        {"torch": tensor_ops, "Path": Path, "TrajDataset": object,
         "Optional": Optional, "Callable": Callable},
    )

    def normalize(actions, enabled):
        times = len(actions)
        fixture.update({
            "states.pth": torch.arange(times, dtype=torch.float32).view(1, times, 1),
            "actions.pth": actions.unsqueeze(0),
            "door_locations.pth": torch.zeros(1, times),
            "wall_locations.pth": torch.zeros(1, times),
            "episode_000.pth": torch.zeros(times, 3, 1, 1, dtype=torch.uint8),
        })
        with redirect_stdout(io.StringIO()):
            dataset = wall_dataset(normalize_action=enabled)
        return dataset[0][1]

    return normalize


def load_le_normalization():
    transforms = SimpleNamespace(WrapTorchTransform=lambda normalizer, **_: normalizer)
    fit, _ = load_definitions(
        UPSTREAM / "le-wm/utils.py", ["get_column_normalizer", "ZScoreNormalizer"],
        {"torch": torch, "np": np, "dt": SimpleNamespace(transforms=transforms)},
    )
    train_forward, = load_definitions(UPSTREAM / "le-wm/train.py", ["lejepa_forward"], {"torch": torch})

    class EncoderInput(Exception):
        pass

    def capture(batch):
        raise EncoderInput(batch["action"])

    def normalize(actions):
        dataset = SimpleNamespace(get_col_data=lambda _: actions.numpy())
        raw = fit(dataset, "action", "action")(actions)
        model = SimpleNamespace(model=SimpleNamespace(encode=capture))
        cfg = SimpleNamespace(history_size=1, num_preds=1, loss=SimpleNamespace(sigreg=SimpleNamespace(weight=1)))
        try:
            train_forward(model, {"action": raw.clone()}, "train", cfg)
        except EncoderInput as result:
            return raw, result.args[0]
        raise RuntimeError("LeWorldModel did not call its encoder")

    return normalize


def load_ling_paths():
    path = UPSTREAM / "lingbot-va/libero_cfg.py"
    cfg_code = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cfg_code.body = [node for node in cfg_code.body if node.lineno >= 33]
    cfg = SimpleNamespace(action_dim=30, env_type="none")
    exec(compile(cfg_code, str(path), "exec"), {"va_libero_cfg": cfg})
    train, server = load_definitions(
        UPSTREAM / "lingbot-va/action_preprocess_excerpt.py", ["_action_post_process", "preprocess_action"],
        {"np": np, "torch": torch, "F": F, "rearrange": rearrange},
    )

    def apply(actions, refit=False):
        # Each latent frame consumes four actions. The dataset inserts a conditioning frame.
        frames = len(actions) // 4 + 1
        q01, q99 = (np.array(cfg.norm_stat[k], dtype=np.float64) for k in ("q01", "q99"))
        values = actions.numpy()
        if refit:
            q01[:values.shape[-1]] = np.quantile(values, 0.01, axis=0)
            q99[:values.shape[-1]] = np.quantile(values, 0.99, axis=0)
        width = len(cfg.used_action_channel_ids)
        raw = np.pad(values, ((0, 0), (0, width - values.shape[-1])))
        train_state = SimpleNamespace(config=cfg, q01=q01[None], q99=q99[None])
        dataset_actions, _ = train(train_state, 0, len(raw), np.arange((frames - 1) * 4 + 1), raw)
        server_state = SimpleNamespace(
            job_config=cfg, action_norm_method=cfg.action_norm_method,
            actions_q01=torch.from_numpy(q01).to(dtype=actions.dtype).view(-1, 1, 1),
            actions_q99=torch.from_numpy(q99).to(dtype=actions.dtype).view(-1, 1, 1),
        )
        history = np.pad(raw, ((4, 0), (0, 0))).T.reshape(width, frames, 4)
        model_actions = server(server_state, history)[0]
        active = cfg.used_action_channel_ids[:values.shape[-1]]
        return tuple(
            stage[active, 1:, :, 0].permute(1, 2, 0).reshape_as(actions) for stage in (dataset_actions, model_actions)
        )

    return cfg, apply


def finite_report(cases, seed):
    dino, le = load_dino_normalization(), load_le_normalization()
    dream_d, = load_definitions(UPSTREAM / "DreamDojo/action_normalizer_excerpt.py", ["Normalizer"], {"torch": torch})
    dream_z, = load_definitions(UPSTREAM / "dreamzero/action_normalizer_excerpt.py", ["Normalizer"], {"torch": torch})
    cfg, ling = load_ling_paths()
    rng = random.Random(seed)
    failures = Counter()
    for label in ("varied", "constant", "near-constant"):
        for _ in range(cases):
            times = rng.choice((4, 8, 12, 16))
            actions = torch.tensor([[rng.uniform(-0.8, 0.8) for _ in range(2)] for _ in range(times)])
            constant = rng.choice((-0.5, 0.0, 0.5))
            if label == "constant":
                actions[:, 1] = constant
            elif label == "near-constant":
                actions[:, 1] = constant + 1e-6 * actions[:, 1]
            le_intermediate, le_encoder = le(actions)
            ling_train, ling_server = ling(actions, refit=True)
            result = {
                "DINO-WM WallDataset default": dino(actions, False),
                "DINO-WM WallDataset with normalization": dino(actions, True),
                "LeWorldModel normalizer (intermediate)": le_intermediate,
                "LeWorldModel encoder input": le_encoder,
                "DreamDojo min_max normalizer": dream_d("min_max", {
                    "min": actions.amin(0).tolist(), "max": actions.amax(0).tolist(),
                }).forward(actions),
                "DreamZero q99 normalizer": dream_z("q99", {
                    "q01": actions.quantile(0.01, dim=0).tolist(), "q99": actions.quantile(0.99, dim=0).tolist(),
                }).forward(actions),
                "LingBot-VA training": ling_train,
                "LingBot-VA server": ling_server,
            }
            for name, actual in result.items():
                assert actual.shape == actions.shape, f"{name}: expected {actions.shape}, got {actual.shape}"
                failures[name, label] += int(not torch.isfinite(actual).all())
    for name in result:
        counts = [f"{cases - failures[name, label]}/{cases}" for label in ("varied", "constant", "near-constant")]
        print(f"{name}: {', '.join(counts)} finite (varied, constant, near-constant)")
    return cfg, ling, any(count for (name, _), count in failures.items() if "intermediate" not in name)


def compare_ling_train_and_server(cfg, ling, cases, seed):
    rng = random.Random(seed)
    q01, q99 = (torch.tensor(cfg.norm_stat[k][:7]) for k in ("q01", "q99"))
    mismatches = Counter()
    for label in ("within", "outside"):
        for case in range(cases):
            raw = (q01 + q99).repeat(rng.choice((4, 8, 12)), 1) / 2
            channel, time = rng.choice((3, 4, 5)), rng.randrange(len(raw))
            factor = rng.uniform(0.05, 0.95) if label == "within" else rng.choice((-0.65, 1.65))
            raw[time, channel] = q01[channel] + factor * (q99[channel] - q01[channel])
            assert raw.abs().amax() <= 1.0
            training, server = ling(raw)
            if not torch.allclose(training, server, atol=2e-6, rtol=0):
                mismatches[label] += 1
                if case == 0:
                    print(f"LingBot-VA {label}: raw {raw[time, channel]:.4f}, "
                          f"training {training[time, channel]:.4f}, server {server[time, channel]:.4f}")
        print(f"LingBot-VA train/server agreement ({label} fitted quantiles): {cases - mismatches[label]}/{cases}")
    return any(mismatches.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=16, help="Case count per input family")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    cfg, ling, numeric_failure = finite_report(args.cases, args.seed)
    parity_failure = compare_ling_train_and_server(cfg, ling, args.cases, args.seed)
    return int(numeric_failure or parity_failure)


if __name__ == "__main__":
    raise SystemExit(main())
