"""Run upstream robotics conditioning checks with tiny CPU inputs.

Pins: LingBot-VA 7c6ffa9, DreamZero ab790c1, Cosmos-Predict2.5 a2c298b.
"""

import argparse
import ast
from enum import Enum
from functools import partial
import math
import random
from types import SimpleNamespace
from typing import List, Optional, Tuple

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import and_masks, or_masks

from wm_fuzzing.properties import assert_batch_isolation
from wm_fuzzing.sources import UPSTREAM

def load_source(path, prepare=None, **bindings):
    path = UPSTREAM / path
    code = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if prepare is not None:
        prepare(code)
    scope = {"torch": torch, "nn": nn, "F": F, "rearrange": rearrange, "math": math, **bindings}
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope


def lingbot_adapter():
    def dense_mask(mask, batch, heads, queries, keys, *, device, _compile):
        row = torch.arange(queries, device=device)[:, None]
        col = torch.arange(keys, device=device)[None, :]
        zero = torch.zeros((), dtype=torch.long, device=device)
        return mask(zero, zero, row, col).float()

    def strip_flex_compile(tree):
        cls = tree.body[0]
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
            "init_mask", "_get_cross_mask_mod", "_get_mask_mod",
        }]
        # CPU builds a small dense mask with the same upstream token predicates.
        init = cls.body[0]
        init.body = init.body[1:]

    scope = load_source("lingbot-va/batch_forward_excerpt.py", strip_flex_compile,
                        and_masks=and_masks, or_masks=or_masks, partial=partial)
    flex = scope["FlexAttnFunc"]
    flex.compiled_create_block_mask = staticmethod(dense_mask)
    model = scope["BatchForwardExcerpt"]()

    def block(hidden, text, *_args, **_kwargs):
        def masked_mean(mask, value):
            return (mask @ value.float()) / mask.sum(1, keepdim=True).clamp_min(1)

        return hidden.float() + masked_mean(flex.attention_mask, hidden) + masked_mean(flex.cross_attention_mask, text)

    def timestep_embedding(steps, dtype):
        return torch.zeros((*steps.shape, 4), dtype=dtype), torch.zeros((*steps.shape, 24), dtype=dtype)

    model.patch_size = (1, 2, 2)
    model.patch_embedding_mlp = nn.Identity()
    model.action_embedder = lambda x: x.expand(-1, -1, 4)
    # The upstream text embedder is a per-token projection.
    class Conditioning:
        text_embedder = nn.Identity()
        __call__ = staticmethod(timestep_embedding)

    model.condition_embedder = model.condition_embedder_action = Conditioning()
    model.rope = lambda grid: torch.zeros(1, grid.shape[-1], 1)
    model.blocks = [block]
    model.scale_shift_table = torch.zeros(1, 2, 4)
    model.norm_out = model.proj_out = model.action_proj_out = nn.Identity()

    def run(batch):
        video, action, text = map(torch.stack, zip(*batch))
        samples, _, frames, _, _ = video.shape
        timesteps = torch.zeros(samples, frames)
        video_grid = torch.zeros(samples, 4, frames)
        action_grid = torch.zeros(samples, 4, frames * action.shape[-2])
        latent_dict = dict(noisy_latents=video, latent=video / 2, text_emb=text,
                           timesteps=timesteps, cond_timesteps=timesteps, grid_id=video_grid)
        action_dict = dict(noisy_latents=action, latent=action / 2, timesteps=timesteps,
                           cond_timesteps=timesteps, grid_id=action_grid)
        video_out, action_out = model.forward_train(dict(latent_dict=latent_dict, action_dict=action_dict,
                                                         chunk_size=2, window_size=4))
        return torch.cat([video_out.flatten(1), action_out.flatten(1)], 1).tolist()

    def make(frames, generator):
        video = torch.randint(-4, 5, (1, frames, 2, 2), generator=generator).float()
        action = torch.randint(-4, 5, (1, frames, 3, 1), generator=generator).float()
        text = torch.randint(-4, 5, (512, 4), generator=generator).float()
        return video, action, text

    return run, make


def dreamzero_adapter():
    class CapturedDiTInputs(Exception):
        pass

    def capture(noisy_video, **kwargs):
        raise CapturedDiTInputs(noisy_video, kwargs["action"], kwargs["state"], kwargs["context"], kwargs["y"])

    forward = load_source("dreamzero/batch_forward_excerpt.py", BatchFeature=object, print=lambda *_: None)["forward"]
    scheduler = load_source("dreamzero/flow_match_scheduler.py")["FlowMatchScheduler"]
    # Callable instance provides the training forward's model metadata and input boundary.
    class DiT:
        num_action_per_block = 4
        num_state_per_block = 1
        __call__ = staticmethod(capture)

    model = SimpleNamespace(
        model=DiT(), config=SimpleNamespace(decouple_video_action_noise=False, use_high_noise_emphasis=False),
        set_frozen_modules_to_eval_mode=lambda: None, normalize_video=lambda x: x * 2 - 1,
        encode_prompt=lambda text, mask: text * mask, encode_video=lambda video, *_: video,
        encode_image=lambda image, *_: (image, image, None),
        dtype=torch.float32, tiled=False, tile_size_height=2, tile_size_width=2,
        tile_stride_height=2, tile_stride_width=2, _device="cpu", _noise_logged=True,
        scheduler=scheduler(num_inference_steps=16, num_train_timesteps=16), num_frame_per_block=2,
    )

    class Data(dict):
        __getattr__ = dict.__getitem__

    def run(batch):
        video, action, state, text = map(torch.stack, zip(*batch))
        inputs = Data(images=video, action=action, state=state, text=text, text_attention_mask=torch.ones_like(text),
                      embodiment_id=torch.zeros(len(batch), dtype=torch.long),
                      has_real_action=torch.ones(len(batch)), action_mask=torch.ones_like(action))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(998)
            try:
                forward(model, None, inputs)
            except CapturedDiTInputs as result:
                return torch.cat([item.flatten(1) for item in result.args], dim=1).tolist()
        raise AssertionError("DreamZero training forward did not reach the DiT boundary")

    def make(frames, generator):
        # Shipped forward requires two latent frames per block for this fixture.
        frames = 2 * frames + 1
        video = torch.randint(0, 256, (frames, 2, 2, 3), generator=generator, dtype=torch.uint8)
        action = torch.rand((2 * (frames - 1), 2), generator=generator) * 2 - 1
        state = torch.rand(((frames - 1) // 2, 3), generator=generator)
        text = torch.rand((4, 3), generator=generator)
        return video, action, state, text

    return run, make


def cosmos_adapter():
    class DataType(Enum):
        VIDEO = "video"

    def expose_action_embeddings(tree):
        forward = tree.body[1].body[0]
        forward.body.extend(ast.parse("return t_embedding_B_T_D, adaln_lora_B_T_3D").body)

    scope = load_source("cosmos_predict2_5/batch_action_excerpt.py", expose_action_embeddings,
                        DataType=DataType, amp=torch.amp, Optional=Optional, List=List, Tuple=Tuple)
    mlp = scope["Mlp"]
    model = scope["BatchActionExcerpt"]()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(892)
        model.action_embedder_B_D = mlp(8, 8, 4).eval()
        model.action_embedder_B_3D = mlp(8, 8, 12).eval()
    model._num_action_per_latent_frame = 4
    model.timestep_scale = 1
    model.use_crossattn_projection = model.use_wan_fp32_strategy = False
    model.prepare_embedded_sequence = lambda video, **_: (video, None, None)
    model.t_embedder = lambda t: (t[..., None].expand(-1, -1, 4), t[..., None].expand(-1, -1, 12))
    model.t_embedding_norm = nn.Identity()

    def run(batch):
        action, condition = map(torch.stack, zip(*batch))
        samples, _, frames, _, _ = condition.shape
        video = torch.zeros(samples, 1, frames, 1, 1)
        context = torch.zeros(samples, 1, 4)
        embedding, modulation = model.forward(video, torch.ones(samples, frames), context,
                                             condition_video_input_mask_B_C_T_H_W=condition, action=action)
        return torch.cat([embedding.flatten(1), modulation.flatten(1)], 1).tolist()

    def make(frames, generator):
        action = torch.rand((4 * frames, 2), generator=generator)
        condition = torch.zeros(1, frames + 1, 1, 1)
        condition[:, :1 + int(torch.randint(0, min(frames, 3), (), generator=generator))] = 1
        return action, condition

    return run, make


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("cases must be positive")
    torch.set_num_threads(1)

    failed = False
    with torch.inference_mode():
        for name, adapter in (("LingBot-VA packed training attention", lingbot_adapter),
                              ("DreamZero training conditioning at DiT input", dreamzero_adapter),
                              ("Cosmos-Predict2.5 action modulation", cosmos_adapter)):
            rng = random.Random(args.seed)
            generator = torch.Generator().manual_seed(args.seed)
            run, make = adapter()
            passed = 0
            for case in range(args.cases):
                count = rng.randint(2, 4)
                frames = rng.randint(2, 5)
                held, changed = rng.sample(range(count), 2)
                batch = [make(frames, generator) for _ in range(count)]
                replacement = make(frames, generator)
                try:
                    assert_batch_isolation(run, batch, replacement, held=held, changed=changed)
                except AssertionError as error:
                    failed = True
                    print(f"{name}, seed={args.seed}, case={case}: {error}")
                else:
                    passed += 1
            print(f"{name}: {passed}/{args.cases} passed", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
