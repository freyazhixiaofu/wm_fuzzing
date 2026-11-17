"""Check independent batched paths in TD-MPC2, DreamerV3, and Dreamer 4.

Pinned upstream sources
  nicklashansen/tdmpc2 332f19e4aba4bffe2dc736f45ea7f7bdc585903a
    tdmpc2/common/buffer.py lines 93–110 and tdmpc2/tdmpc2.py lines 241–257.
  danijar/dreamerv3 e01491fad6434b2245a3b8ca201dd7faedcc458c
    dreamerv3/rssm.py lines 45–92.
  nicklashansen/dreamer4 b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6
    dreamer4/model.py.
"""

import argparse
import random
import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from fuzz_checks import assert_batch_isolation
from fuzz_tdmpc2 import CpuBatch, load_method


SOURCE = Path(__file__).with_name("upstream") / "batch_latent_more"


def load_tdmpc2():
    prepare = load_method(SOURCE / "tdmpc2-buffer-current.py", "Buffer", "_prepare_batch")
    target = load_method(SOURCE / "tdmpc2-current.py", "TDMPC2", "_td_target")
    buffer = SimpleNamespace(_device="cpu")
    heads = SimpleNamespace(
        pi=lambda observations, task: (observations * 0.5, None),
        Q=lambda observations, action, task, **_: observations + action,
    )
    model = SimpleNamespace(
        model=heads, discount=torch.tensor([0.25, 0.5, 0.75]),
        cfg=SimpleNamespace(multitask=True),
    )

    def run(samples):
        tags = torch.tensor(samples, dtype=torch.float32).T.contiguous()
        task = tags[0].to(torch.long).remainder(3).expand_as(tags)
        batch = CpuBatch(
            obs=tags.unsqueeze(-1), action=tags.neg().unsqueeze(-1),
            reward=2 * tags, terminated=(tags.to(torch.long) % 4 == 0).float(),
            task=task,
        )
        obs, action, reward, terminal, task = prepare(buffer, batch)
        td = target(model, obs[1:], reward, terminal, task)
        # The sampled tensors are time-major. Return one result per sample.
        return torch.cat((obs[1:], action, terminal, td), dim=-1).transpose(0, 1).tolist()

    return run


def load_dreamerv3():
    def mask(values, valid):
        def masked(x):
            return np.where(valid.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0)
        return tuple(map(masked, values)) if isinstance(values, tuple) else masked(values)

    linear = object()
    norm = object()
    scope = {
        "jnp": np, "f32": np.float32,
        "nn": SimpleNamespace(
            cast=lambda x: x, mask=mask, DictConcat=lambda *_: lambda x: x,
            Linear=linear, Norm=norm, act=lambda _: np.tanh, COMPUTE_DTYPE=np.float32,
        ),
        "nj": SimpleNamespace(seed=lambda: 0),
        "jax": SimpleNamespace(tree=SimpleNamespace(
            leaves=lambda d: list(d.values()),
            map=lambda fn, d: {k: fn(x) for k, x in d.items()},
        )),
    }
    source = SOURCE / "dreamerv3-rssm.py"
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), scope)
    rssm = scope["RSSM"]

    def sub(_name, kind, *_args, **_kwargs):
        if kind is linear:
            return lambda x: x.sum(axis=-1, keepdims=True)
        return lambda x: x

    def run(samples):
        tags = np.array(samples, dtype=np.float32)
        model = SimpleNamespace(
            deter=1, stoch=1, classes=1, act_space=None,
            absolute=False, obslayers=1, hidden=1, kw={}, act=None, norm=None, sub=sub,
            _core=lambda deter, stoch, action: 0.5 * deter + stoch[:, 0] + action,
            _logit=lambda _name, x: x[:, None],
            _dist=lambda x: SimpleNamespace(sample=lambda **_: x),
        )
        model._observe = MethodType(rssm._observe, model)
        carry = rssm.initial(model, len(tags))
        observed = []
        for step in tags.T:
            carry, entry, _ = rssm.observe(
                model, carry, step[:, None] * 0.125, step[:, None] * 0.25,
                step.astype(np.int32) % 4 == 0, training=False, single=True,
            )
            observed.append(entry)

        history = {key: np.stack([entry[key] for entry in observed], axis=1) for key in carry}
        # Keep every imagined start in its original row when interpreting the flattened output.
        starts = rssm.starts(model, history, carry, nlast=tags.shape[1])
        output = np.concatenate([x.reshape(len(tags), tags.shape[1], -1) for x in starts.values()], axis=-1)
        return output.tolist()

    return run


def load_dreamer4():
    module = runpy.run_path(str(SOURCE / "dreamer4-model.py"))
    model = module["Dynamics"](
        d_model=16, d_bottleneck=2, d_spatial=4, n_spatial=2,
        n_register=1, n_agent=1, n_heads=2, depth=2, k_max=4,
        mlp_ratio=2, time_every=1,
    ).eval()
    # Observe the spatial output as well as hidden tokens. The stock initializer zeros this head.
    with torch.no_grad():
        model.flow_x_head.weight.normal_(std=0.1)

    def run(samples):
        tags = torch.tensor(samples, dtype=torch.float32)
        actions = torch.sin(tags[..., None] * 0.125 + torch.arange(16))
        packed = torch.cos(tags[..., None] * 0.25 + torch.arange(8)).reshape(*tags.shape, 2, 4)
        step = tags.to(torch.long).remainder(3)
        signal = tags.to(torch.long).remainder(5)
        spatial, agent = model(actions, step, signal, packed)
        return torch.cat((spatial.flatten(2), agent.flatten(2)), dim=-1).tolist()

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=48)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("cases must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cases = []
    for _ in range(args.cases):
        size, steps = rng.randint(2, 5), rng.randint(2, 7)
        batch = [[rng.randrange(-32, 33) for _ in range(steps)] for _ in range(size)]
        held, changed = rng.sample(range(size), 2)
        replacement = [value + rng.randrange(9, 20) for value in batch[changed]]
        cases.append((batch, replacement, held, changed))

    failures = 0
    with torch.inference_mode():
        for name, load in (("TD-MPC2", load_tdmpc2), ("DreamerV3", load_dreamerv3), ("Dreamer 4", load_dreamer4)):
            run = load()
            batch, replacement, _, changed = cases[0]
            updated = list(batch)
            updated[changed] = replacement
            if run(batch)[changed] == run(updated)[changed]:
                raise AssertionError(f"{name}: changed-sample control left its own output unchanged")
            failed = 0
            for case, (batch, replacement, held, changed) in enumerate(cases):
                try:
                    assert_batch_isolation(run, batch, replacement, held=held, changed=changed)
                except AssertionError as error:
                    if failed == 0:
                        print(f"{name} seed={args.seed} case={case}: {error}")
                    failed += 1
            failures += failed
            print(f"{name}: {len(cases) - failed}/{len(cases)} batch isolation cases passed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
