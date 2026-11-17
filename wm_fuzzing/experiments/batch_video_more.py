"""Check batch isolation in CPU-compatible components of three video world models."""

import argparse
import ast
import json
import random
import runpy
from types import MethodType, SimpleNamespace
from typing import List, Optional

import torch
from einops import rearrange

from wm_fuzzing.experiments.hy_vae_overlap import load_decode_methods
from wm_fuzzing.properties import assert_batch_isolation
from wm_fuzzing.sources import UPSTREAM

def hy_worldplay():
    # Tencent-Hunyuan/HY-WorldPlay, 1588e1336e842b03b0a7860c654ebd7c46bb065e.
    source = UPSTREAM / "hy_worldplay"
    config = json.loads((source / "fasthunyuan_t2v.json").read_text(encoding="utf8"))["vae_config"]
    methods = load_decode_methods(source / "temporal_vae_common.py")

    def decode_tile(self, tile):
        # Keep the tile's conditioning frame and emit four frames for each later latent.
        return torch.cat((tile[:, :, :1], tile[:, :, 1:].repeat_interleave(4, dim=2)), dim=2)

    methods["_decode"] = decode_tile
    model = type("NumberedVAE", (), methods)()
    model.__dict__.update(config)
    model.temporal_compression_ratio = 4
    model.spatial_compression_ratio = 8

    def run(batch):
        z = torch.tensor(batch, dtype=torch.float32)[:, None, :, None, None]
        return model.decode(z).flatten(1).tolist()

    return run


def vista():
    # OpenDriveLab/Vista, cc9821b4253ca7987c32757613d2fc2448fa9f5d,
    # vwm/modules/diffusionmodules/loss.py.
    path = UPSTREAM / "Vista" / "loss.py"
    module = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    loss_class = next(node for node in module.body if getattr(node, "name", None) == "StandardDiffusionLoss")
    methods = [node for node in loss_class.body if getattr(node, "name", None) in {"_forward", "get_noised_input"}]
    scope = {
        "torch": torch,
        "rearrange": rearrange,
        "append_dims": lambda value, ndim: value[(...,) + (None,) * (ndim - value.ndim)],
        "Denoiser": object,
        "Dict": dict,
        "nn": torch.nn,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)

    def run(batch):
        size, frames = len(batch), len(batch[0])
        clean = torch.tensor(batch, dtype=torch.float32).reshape(size * frames, 1, 1, 1)
        model = SimpleNamespace(
            num_frames=frames,
            cond_frames_choices=[[], [0], [0, 1]],
            replace_cond_frames=True,
            offset_noise_level=0.125,
            sigma_sampler=lambda count: torch.ones(count),
            loss_weighting=torch.ones_like,
        )
        model.get_noised_input = MethodType(scope["get_noised_input"], model)
        model.get_loss = lambda prediction, target, weights: prediction
        scope["random"] = random.Random(77)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            # Capture the actual masked prediction/target selection before reducing the training loss.
            result = scope["_forward"](model, None, lambda _, noised, *args: noised * 2, {}, clean)
        return result.reshape(size, -1).tolist()

    return run


def abot_world():
    source = UPSTREAM / "reset" / "abot"
    common = {"torch": torch, "Optional": Optional, "List": List, "print": lambda *args, **kwargs: None}
    cache = runpy.run_path(str(source / "stream.py"), init_globals=common)
    methods = runpy.run_path(
        str(source / "batched_inference.py"),
        init_globals={**common, "tqdm": SimpleNamespace(tqdm=lambda items: items)},
    )

    def run(batch):
        noise = torch.tensor(batch, dtype=torch.float32)[..., None, None, None] * 0.01
        initial = noise[:, :4] * -0.5
        action = noise.transpose(1, 2).contiguous()

        def generator(*, noisy_image_or_video, conditional_dict, timestep, **kwargs):
            tags = conditional_dict["act_context"].transpose(1, 2)
            return None, noisy_image_or_video + tags * 0.125

        generator.model = SimpleNamespace(num_heads=1, dim=2)
        model = SimpleNamespace(
            generator=generator,
            scheduler=SimpleNamespace(add_noise=lambda pred, perturbation, _: pred + perturbation * 0.01),
            local_attn_size=4,
            frame_seq_length=2,
            num_transformer_blocks=1,
            args=SimpleNamespace(context_noise=0),
            kv_cache1=None,
            crossattn_cache=None,
            num_frame_per_block=4,
            independent_first_frame=False,
            denoising_step_list=[10, 5],
            _decode_output_to_video=lambda latents: latents,
        )
        for name in ("_ref_cache_token_len", "_kv_cache_needs_reinit", "_initialize_kv_cache", "_initialize_crossattn_cache"):
            setattr(model, name, MethodType(cache[name], model))
        model.get_condition_split = MethodType(methods["get_condition_split"], model)
        conditions = {"prompt_embeds": torch.zeros(len(batch), 1, 2), "act_context": action}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            _, latents = methods["inference"](model, noise, [""] * len(batch), conditions, initial, return_latents=True)
        return latents.flatten(1).tolist()

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    torch.set_num_threads(1)
    failed = False
    adapters = (
        ("HY-WorldPlay temporal VAE tiling", hy_worldplay()),
        ("Vista training conditioning mask", vista()),
        ("ABot-World offline latent assembly", abot_world()),
    )
    for name, run in adapters:
        rng = random.Random(args.seed)
        passed = 0
        for case in range(args.cases):
            size = 2 + case % 3
            frames = 4 * rng.randint(2, 5)
            batch = [[rng.randrange(-40, 41) for _ in range(frames)] for _ in range(size)]
            held, changed = rng.sample(range(size), 2)
            replacement = [value + 41 for value in batch[changed]]
            try:
                with torch.inference_mode():
                    assert_batch_isolation(run, batch, replacement, held=held, changed=changed)
            except AssertionError as error:
                if case == passed:
                    print(f"{name} case {case}, batch {size}: {error}")
                failed = True
            else:
                passed += 1
        print(f"{name}: {passed}/{args.cases} passed")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
