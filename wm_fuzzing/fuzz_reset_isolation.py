"""Exercise upstream CPU reset and resume paths without learned components."""

import argparse
import random
import sys
from collections import namedtuple
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


SOURCE = Path(__file__).with_name("upstream") / "reset"


def load_source(relative_path, **bindings):
    source = SOURCE / relative_path
    compiled = compile("from __future__ import annotations\n" + source.read_text(encoding="utf-8"), str(source), "exec")
    scope = {"np": np, "torch": torch, **bindings}
    exec(compiled, scope)
    return scope


def bind(instance, scope, *names):
    for name in names:
        setattr(instance, name, MethodType(scope[name], instance))
    return instance


def check_reset_isolation(make, reset, warm, probe):
    """A used instance must behave like a fresh one after the same reset."""
    fresh, used = make(), make()
    reset(fresh)
    reset(used)
    warm(used)
    reset(used)
    return probe(fresh), probe(used)


class TinyEnv:
    def __init__(self, horizon):
        self.horizon = horizon
        self.position = 0

    def step(self, action):
        self.position = 0 if action["reset"] else self.position + 1
        done = self.position == self.horizon
        return dict(position=self.position, reward=float(action["value"]), is_last=done, is_terminal=done)


def check_action_repeat(wrapper, rng):
    repeat = rng.randint(2, 6)
    horizon = rng.randint(3 * repeat + 1, 7 * repeat)

    def warm(instance):
        while not instance.step(dict(reset=False, value=-2))["is_last"]:
            pass

    return check_reset_isolation(
        lambda: wrapper(TinyEnv(horizon), repeat),
        lambda instance: instance.step(dict(reset=True, value=0)),
        warm,
        lambda instance: [instance.step(dict(reset=False, value=1)) for _ in range(3)],
    )


def diamond_init(count):
    obs = torch.arange(3).repeat(count, 1)
    actions = torch.zeros_like(obs)
    zeros = torch.zeros(1, count, 1)
    return obs, actions, (zeros.clone(), zeros.clone())


def check_diamond(methods, rng):
    batch = rng.randint(2, 5)
    warm_steps = rng.randint(1, 8)
    horizon = rng.randint(1, 6)

    def make():
        model = SimpleNamespace(
            generator_init=SimpleNamespace(send=diamond_init), horizon=horizon,
            num_envs=batch, return_denoising_trajectory=False,
        )

        def predict_reward(_):
            model.hx_rew_end += 1
            model.cx_rew_end += 3
            return model.act_buffer[:, -1].clone(), torch.zeros(batch, dtype=torch.bool)

        model.predict_next_obs = lambda: (model.obs_buffer[:, -1] + model.act_buffer[:, -1], [])
        model.predict_rew_end = predict_reward
        return bind(model, methods, "reset", "reset_dead", "step")

    def run(model, actions):
        trajectory = []
        for action in actions:
            outputs = model.step(torch.tensor(action))
            trajectory.append(tuple(t.tolist() for t in outputs[:4]))
        return trajectory

    warm_actions = [[rng.randint(1, 7) for _ in range(batch)] for _ in range(warm_steps)]
    probe_actions = [[rng.randint(-4, 4) for _ in range(batch)] for _ in range(4)]
    full = check_reset_isolation(make, lambda model: model.reset(), lambda model: run(model, warm_actions),
                                 lambda model: run(model, probe_actions))

    model = make()
    model.reset()
    run(model, warm_actions)
    dead = torch.tensor([(i + warm_steps) % 2 == 0 for i in range(batch)])
    fields = ("obs_buffer", "act_buffer", "hx_rew_end", "cx_rew_end", "ep_len")
    batch_axes = (0, 0, 1, 1, 0)
    fresh = make()
    fresh.reset()
    expected = []
    for name, axis in zip(fields, batch_axes):
        value = getattr(model, name).movedim(axis, 0).clone()
        value[dead] = getattr(fresh, name).movedim(axis, 0)[dead]
        expected.append(value.tolist())
    model.reset_dead(dead)
    actual = [getattr(model, name).movedim(axis, 0).tolist() for name, axis in zip(fields, batch_axes)]
    return full, (expected, actual)


def check_overworld(engine, codec, taehv, rng):
    warm_values = [rng.randint(80, 127) / 128 for _ in range(rng.randint(1, 6))]
    probe_values = [rng.randint(1, 47) / 128 for _ in range(3)]
    # Only the tiny deterministic network weights stand in. The memory block,
    # streaming queue, and engine-level adapter are upstream implementations.
    memory = taehv["MemBlock"](3, 3)
    with torch.no_grad():
        for layer in (memory.conv[0], memory.conv[2], memory.conv[4]):
            layer.weight.zero_()
            layer.bias.zero_()
        for channel in range(3):
            memory.conv[0].weight[channel, channel + 3, 1, 1] = 0.5
            memory.conv[2].weight[channel, channel, 1, 1] = 1
            memory.conv[4].weight[channel, channel, 1, 1] = 1
    weights = SimpleNamespace(encoder=(), decoder=(memory,), latent_channels=3, is_cogvideox=False,
                              frames_to_trim=0, postprocess_output_frames=lambda frame: frame)
    streaming = taehv["StreamingTAEHV"]

    def make():
        vae = SimpleNamespace(
            streaming_ae_model=streaming(weights),
            auto_aspect_ratio=False, device="cpu", dtype=torch.float32,
        )
        bind(vae, codec, "reset", "decode")
        model = SimpleNamespace(
            vae=vae, frame_ts=torch.zeros(1, dtype=torch.long), _ctx={"value": torch.zeros(1)},
            frm_shape=(1, 1, 3, 1, 1), device="cpu", dtype=torch.float32,
        )
        cache = SimpleNamespace(position=torch.zeros(1))
        cache.reset = cache.position.zero_
        cache.get_state = cache.position.clone
        cache.load_state = cache.position.copy_
        model.kv_cache = cache

        def prep(x, ctrl):
            model.frame_ts += 1
            return ctrl

        model.prep_inputs = prep
        model._denoise_pass = lambda latent, value, _: torch.full_like(latent, value)
        model._cache_pass = lambda latent, value, cache: cache.position.add_(1)
        return bind(model, engine, "reset", "gen_frame", "get_state", "load_state")

    def run(instance, values):
        return [instance.gen_frame(value).flatten().tolist() for value in values]

    with patch.dict(sys.modules, {"taehv": SimpleNamespace(StreamingTAEHV=streaming)}):
        reset = check_reset_isolation(make, lambda model: model.reset(), lambda model: run(model, warm_values),
                                      lambda model: run(model, probe_values))
        model = make()
        run(model, warm_values)
        snapshot = model.get_state()
        expected = run(model, probe_values)
        model.load_state(snapshot)
        actual = run(model, probe_values)
    return reset, (expected, actual)


def check_abot(methods, rng, resize):
    start_batch = rng.randint(1, 4)
    batch = start_batch + rng.choice((1, 2)) if resize else start_batch
    history = rng.randint(1, 4)

    def make():
        model = SimpleNamespace(
            kv_cache1=None, crossattn_cache=None, conditional_dict=None, local_attn_size=4,
            frame_seq_length=2, num_transformer_blocks=2,
            generator=SimpleNamespace(model=SimpleNamespace(num_heads=2, dim=6)),
        )
        return bind(model, methods, "reset_stream", "_ref_cache_token_len", "_kv_cache_needs_reinit",
                    "_initialize_kv_cache", "_initialize_crossattn_cache")

    def reset(model, size):
        model.reset_stream(size, torch.float32, "cpu", torch.empty(size, history, 1))

    def warm(model):
        reset(model, start_batch)
        model.current_start_frame = 9
        model._stream_block_decode_times.append(1.0)
        model._stream_block_diffusion_times.append(1.0)
        for cache, cross in zip(model.kv_cache1, model.crossattn_cache):
            cache["global_end_index"].fill_(7)
            cache["local_end_index"].fill_(3)
            cross["is_init"] = True

    def probe(model):
        return (
            model.current_start_frame, model.num_input_frames,
            model._stream_block_decode_times, model._stream_block_diffusion_times,
            [(cache["k"].size(0), cache["global_end_index"].item(), cache["local_end_index"].item(),
              cross["k"].size(0), cross["is_init"]) for cache, cross in zip(model.kv_cache1, model.crossattn_cache)],
        )

    fresh, used = make(), make()
    reset(fresh, batch)
    warm(used)
    reset(used, batch)
    return probe(fresh), probe(used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    # The old wrapper only needs its base constructor to keep a reference to the environment.
    class Wrapper:
        def __init__(self, env):
            self.env = env

    wrappers = {
        version: load_source(f"dreamerv3/{version}.py", base=SimpleNamespace(Wrapper=Wrapper))["ActionRepeat"]
        for version in ("before", "fixed", "current")
    }
    diamond = load_source("diamond/episode.py")
    overworld = load_source("overworld/engine.py")
    codec = load_source("overworld/ae.py")
    taehv = load_source("overworld/taehv.py", nn=torch.nn, namedtuple=namedtuple)
    abot = load_source("abot/stream.py", print=lambda *args, **kwargs: None)
    torch.set_num_threads(1)

    results = {}
    for case in range(args.cases):
        case_seed = args.seed + case
        sample = {
            f"DreamerV3 action repeat {version}": check_action_repeat(wrapper, random.Random(case_seed))
            for version, wrapper in wrappers.items()
        }
        sample["DIAMOND full reset"], sample["DIAMOND selective reset"] = check_diamond(diamond, random.Random(case_seed))
        sample["Overworld reset"], sample["Overworld snapshot replay"] = check_overworld(
            overworld, codec, taehv, random.Random(case_seed)
        )
        sample["ABot-World same-batch reset"] = check_abot(abot, random.Random(case_seed), resize=False)
        sample["ABot-World resized-batch reset"] = check_abot(abot, random.Random(case_seed), resize=True)
        for name, (expected, actual) in sample.items():
            results.setdefault(name, []).append(expected == actual)
            if expected != actual and all(results[name][:-1]):
                print(f"{name}, seed {case_seed}: expected {expected}; got {actual}")

    for name, matches in results.items():
        print(f"{name}: {sum(matches)}/{args.cases} matched")
    return int(any(not all(matches) for name, matches in results.items() if not name.endswith(" before")))


if __name__ == "__main__":
    raise SystemExit(main())
