"""Fuzz TD-MPC2's termination-mask alignment on CPU using upstream methods."""

import argparse
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import torch


from wm_fuzzing.sources import UPSTREAM


SNAPSHOTS = UPSTREAM / "tdmpc2"


class CpuBatch(dict):
    """The subset of TensorDict used by the sampling methods under test."""

    def select(self, *keys, strict):
        return type(self)({key: self[key] for key in keys if key in self})

    def to(self, device, non_blocking=False):
        assert torch.device(device).type == "cpu"
        return self


class FixedValueModel:
    def pi(self, states, task):
        return None, None

    def Q(self, states, actions, task, *, return_type, target):
        return torch.ones_like(states)


def load_method(path, class_name, method_name):
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for cls in source.body:
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    scope = {"torch": torch}
                    compiled = compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec")
                    exec(compiled, scope)
                    return scope[method_name]
    raise ValueError(f"Could not find {class_name}.{method_name} in {path}")


def check_alignment(buffer_source, model_source, *, seed, cases):
    prepare_batch = load_method(buffer_source, "Buffer", "_prepare_batch")
    td_target = load_method(model_source, "TDMPC2", "_td_target")
    buffer = SimpleNamespace(_device="cpu", _to_device=lambda *values: values)
    model = SimpleNamespace(model=FixedValueModel(), discount=0.5, cfg=SimpleNamespace(multitask=False))
    rng = random.Random(seed)
    failures = 0
    for case in range(cases):
        if case == 0:
            horizon, batch_size = 3, 1
        else:
            horizon, batch_size = rng.randint(2, 12), rng.randint(1, 5)

        terminal = torch.zeros(horizon + 1, batch_size)
        terminal[-1] = torch.tensor([1] + [rng.randint(0, 1) for _ in range(batch_size - 1)])
        zeros = torch.zeros_like(terminal)
        batch = CpuBatch(obs=zeros.unsqueeze(-1), action=zeros.unsqueeze(-1), reward=zeros, terminated=terminal)
        obs, _, reward, mask, task = prepare_batch(buffer, batch)
        actual = td_target(model, obs[1:], reward, mask, task)
        expected = (0.5 * (1 - terminal[1:])).unsqueeze(-1)
        if not torch.equal(actual, expected):
            if failures == 0:
                print(
                    f"  seed={seed}, case={case}: first trajectory expected "
                    f"{expected[:, 0, 0].tolist()}, received {actual[:, 0, 0].tolist()}"
                )
            failures += 1
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, nargs="?", help="Local TD-MPC2 source checkout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=32)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")

    if args.checkout:
        try:
            failures = check_alignment(
                args.checkout / "tdmpc2" / "common" / "buffer.py",
                args.checkout / "tdmpc2" / "tdmpc2.py",
                seed=args.seed,
                cases=args.cases,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"Local checkout: {args.cases - failures}/{args.cases} passed")
        return int(failures != 0)

    expected_failures = {"3789fcd": args.cases, "add30b5": 0, "eece801": 0}
    unexpected_result = False
    for ref, expected in expected_failures.items():
        snapshot = SNAPSHOTS / f"{ref}.py"
        failures = check_alignment(snapshot, snapshot, seed=args.seed, cases=args.cases)
        print(f"TD-MPC2 {ref}: {args.cases - failures}/{args.cases} passed")
        unexpected_result |= failures != expected
    return int(unexpected_result)


if __name__ == "__main__":
    raise SystemExit(main())
