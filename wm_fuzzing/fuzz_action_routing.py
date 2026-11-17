"""Check that an action pulse reaches the intended history slot or prediction step.

Snapshots: DIAMOND 5bcd1599755b4f2fae8e5e079e02f0728e174965,
Matrix-Game-2 51913b2bfa78a6d54de912b8348b1d318f4d36e2, and
V-JEPA 2-AC 204698b45b3712590f06245fbfba32d3be539812.
"""

import __future__
import argparse
import ast
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SOURCE = Path(__file__).with_name("upstream") / "action_routing"


def load_nodes(path, scope, select):
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    tree.body = [node for node in tree.body if select(node)]
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), scope)


def load_diamond():
    scope = {"torch": torch, "Categorical": torch.distributions.Categorical}
    path = SOURCE / "diamond" / "world_model_env.py"
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    model = next(node for node in tree.body if getattr(node, "name", None) == "WorldModelEnv")
    methods = {"step", "predict_next_obs", "predict_rew_end"}
    model.body = [node for node in model.body if getattr(node, "name", None) in methods]
    tree.body = [model]
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope["WorldModelEnv"]


def check_diamond(model, rng):
    batch = rng.randint(1, 4)
    history_size = rng.randint(2, 8)
    steps = history_size + rng.randint(2, 7)
    impulse_step, sample = rng.randrange(steps), rng.randrange(batch)
    action_history = torch.zeros(batch, steps, dtype=torch.long)
    action_history[sample, impulse_step] = 3
    observed_windows = []
    reward_actions = []

    def sample_next(observations, actions):
        observed_windows.append(actions.clone())
        return observations[:, -1] + actions[:, -1:], []

    def predict_reward(obs, actions, next_obs, recurrent_state):
        reward_actions.append(actions.clone())
        reward_logits = torch.tensor([[[-float("inf"), 0.0, -float("inf")]]]).expand(batch, -1, -1)
        end_logits = torch.tensor([[[0.0, -float("inf")]]]).expand(batch, -1, -1)
        return reward_logits, end_logits, recurrent_state

    env = model()
    env.obs_buffer = torch.zeros((batch, history_size, 1))
    env.act_buffer = torch.zeros((batch, history_size), dtype=torch.long)
    env.ep_len = torch.zeros(batch, dtype=torch.long)
    env.hx_rew_end = env.cx_rew_end = None
    env.horizon = steps + 1
    env.return_denoising_trajectory = False
    env.sampler = SimpleNamespace(sample=sample_next)
    env.rew_end_model = SimpleNamespace(predict_rew_end=predict_reward)

    padding = torch.zeros(batch, history_size - 1, dtype=torch.long)
    for step in range(steps):
        obs, *_ = env.step(action_history[:, step])
        expected = torch.cat((padding, action_history[:, :step + 1]), dim=1)
        torch.testing.assert_close(observed_windows[-1], expected[:, -history_size:])
        torch.testing.assert_close(reward_actions[-1], action_history[:, step:step + 1])
        torch.testing.assert_close(obs.squeeze(1), action_history[:, :step + 1].sum(1).float())


class CaptureProjection(Exception):
    pass


def capture_projection(value):
    raise CaptureProjection(value)


def load_matrix():
    import einops

    scope = {"torch": torch, "nn": nn, "rearrange": einops.rearrange}
    path = SOURCE / "matrix_game2"
    load_nodes(path / "action_module.py", scope, lambda node: getattr(node, "name", None) == "ActionModule")
    load_nodes(path / "cond_current.py", scope, lambda node: getattr(node, "name", None) == "cond_current")
    return scope["cond_current"], scope["ActionModule"].forward


def check_matrix(model, rng):
    cond_current, forward = model
    frames_per_block, compression, history = 3, 4, 3
    latent_frames = frames_per_block * rng.randint(2, 6)
    video_frames = compression * (latent_frames - 1) + 1
    first_chunk_boundary = compression * (frames_per_block - 1)
    impulse_step = rng.choice((
        0, 1, first_chunk_boundary - 1, first_chunk_boundary,
        first_chunk_boundary + 1, video_frames - 1, rng.randrange(video_frames),
    ))
    mode = rng.choice(("universal", "templerun"))
    axis = rng.choice(("keyboard", "mouse")) if mode == "universal" else "keyboard"
    channels = 2 if axis == "mouse" else 6
    impulse_channel = rng.randrange(channels)
    conditioning = {
        "cond_concat": torch.zeros(1, 1, latent_frames),
        "visual_context": None,
        "keyboard_cond": torch.zeros(1, video_frames, 6),
        "mouse_cond": torch.zeros(1, video_frames, 2),
    }
    conditioning[f"{axis}_cond"][0, impulse_step, impulse_channel] = 1
    model = SimpleNamespace(
        vae_time_compression_ratio=compression, windows_size=history,
        enable_mouse=(axis == "mouse"), enable_keyboard=(axis == "keyboard"),
        freqs_cos=None, freqs_sin=None, mouse_mlp=capture_projection,
        mouse_attn_q=lambda states: states, keyboard_embed=lambda inputs: inputs,
        keyboard_attn_kv=capture_projection,
    )
    for start in range(0, latent_frames, frames_per_block):
        cond = cond_current(conditioning, start, frames_per_block, mode=mode)
        limit = compression * (start + frames_per_block - 1) + 1
        assert cond[f"{axis}_cond"].shape[1] == limit
        try:
            forward(
                model, torch.zeros(1, frames_per_block, 1), frames_per_block, 1, 1,
                mouse_condition=cond.get("mouse_cond"), keyboard_condition=cond["keyboard_cond"],
                is_causal=True, kv_cache_mouse={}, kv_cache_keyboard={}, start_frame=start,
                num_frame_per_block=frames_per_block,
            )
        except CaptureProjection as captured:
            window_width = compression * history
            actual = captured.args[0][:, :, -window_width * channels:]
            actual = actual.reshape(frames_per_block, window_width, channels)
        else:
            raise AssertionError("action projection was not reached")
        expected = torch.zeros_like(actual)
        for relative in range(frames_per_block):
            end = compression * (start + relative)
            for slot, index in enumerate(range(end - window_width, end)):
                if max(index, 0) == impulse_step:
                    expected[relative, slot, impulse_channel] = 1
        torch.testing.assert_close(actual, expected)


def load_vjepa():
    import logging
    import math
    from functools import partial

    scope = {
        "torch": torch, "nn": nn, "F": nn.functional,
        "math": math, "partial": partial, "getLogger": logging.getLogger,
    }
    path = SOURCE / "vjepa2"

    def definitions(node):
        return isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))

    # The default predictor has zero stochastic depth. No timm drop_path call is made.
    load_nodes(path / "modules.py", scope, definitions)
    load_nodes(path / "tensors.py", scope, definitions)
    scope["Block"] = scope["ACBlock"]
    load_nodes(path / "ac_predictor.py", scope, definitions)
    return scope["VisionTransformerPredictorAC"]


def make_vjepa(model, steps, grid=1, extrinsics=False, causal=True):
    return model(img_size=16 * grid, num_frames=2 * steps, embed_dim=16,
                 predictor_embed_dim=48, depth=2, num_heads=2,
                 use_extrinsics=extrinsics, is_frame_causal=causal).eval()


def check_vjepa(model_type, rng):
    batch, steps, grid = rng.randint(1, 3), rng.randint(2, 7), rng.randint(1, 2)
    extrinsics = rng.choice((True, False))
    predictor = make_vjepa(model_type, steps, grid, extrinsics)
    frame_tokens = grid ** 2
    x = torch.randn(batch, steps * frame_tokens, 16)
    actions = torch.randn(batch, steps, 7)
    states = torch.randn_like(actions)
    camera = torch.randn(batch, steps, 6) if extrinsics else None
    step, sample, channel = rng.randrange(steps), rng.randrange(batch), rng.randrange(7)
    baseline = predictor(x, actions, states, camera)
    changed_actions = actions.clone()
    changed_actions[sample, step, channel] += 32
    changed = predictor(x, changed_actions, states, camera)
    begin, end = step * frame_tokens, (step + 1) * frame_tokens
    torch.testing.assert_close(changed[sample, :begin], baseline[sample, :begin], rtol=0, atol=0)
    if torch.allclose(changed[sample, begin:end], baseline[sample, begin:end], atol=1e-6, rtol=0):
        raise AssertionError(f"action pulse missing at its destination frame {step}")


def check_vjepa_noncausal(model_type):
    predictor = make_vjepa(model_type, steps=2, causal=False)
    predictor(torch.zeros(1, 2, 16), torch.zeros(1, 2, 7), torch.zeros(1, 2, 7))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=64)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    torch.set_num_threads(1)
    checks = [
        ("DIAMOND action history and reward input", load_diamond(), check_diamond),
        ("Matrix-Game-2 cached action windows", load_matrix(), check_matrix),
        ("V-JEPA 2-AC causal action routing", load_vjepa(), check_vjepa),
    ]
    failed = False
    with torch.inference_mode():
        for name, model, check in checks:
            passed = 0
            first_error = None
            for case in range(args.cases):
                seed = args.seed + case
                torch.manual_seed(seed)
                try:
                    check(model, random.Random(seed))
                except Exception as error:
                    if first_error is None:
                        first_error = f"{name} seed={seed}: {type(error).__name__}: {error}"
                    failed = True
                else:
                    passed += 1
            print(f"{name}: {passed}/{args.cases} passed")
            if first_error is not None:
                print(first_error)
        try:
            check_vjepa_noncausal(checks[-1][1])
        except Exception as error:
            print(f"V-JEPA 2-AC optional noncausal configuration: {type(error).__name__}: {error}")
            failed = True
        else:
            print("V-JEPA 2-AC optional noncausal configuration passed")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
