"""Check Matrix-Game-4 frame assembly and HY-WorldPlay pose timing on CPU.

The runner executes selected upstream functions and the Matrix-Game render loop
with numbered frames. Use the bundled source snapshots or pass local source paths.
"""

import argparse
import ast
import random
import time
from pathlib import Path
from types import SimpleNamespace

from wm_fuzzing.experiments.checks import assert_frame_order
from wm_fuzzing.sources import UPSTREAM


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def run_code(path, statements, bindings):
    module = ast.Module(body=list(statements), type_ignores=[])
    exec(compile(module, str(path), "exec"), bindings)


def check_matrix_game(path, seed):
    import numpy as np

    tree = parse(path)
    frame_blocks = next(node for node in tree.body if getattr(node, "name", "") == "frame_blocks")
    main = next(node for node in tree.body if getattr(node, "name", "") == "main")
    render_loop = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.For) and "frame_blocks(reader)" in ast.unparse(node.iter)
    )

    rng = random.Random(seed)
    lengths = [1, 2, 4, 5, 6, 8, 9, 12, 13] + [rng.randrange(1, 97) for _ in range(32)]
    for length in lengths:
        expected = list(range(length))
        starts, output = [], []

        def push(frames, *, frame_start):
            starts.append(frame_start)
            return SimpleNamespace(pixels=frames)

        bindings = {
            "reader": iter(bytes((value, 0, 0)) for value in expected),
            "caption_file": None,
            "crop_and_resize": lambda frame, *_: frame,
            "Image": SimpleNamespace(fromarray=lambda frame: frame),
            "np": np,
            "width": 1,
            "height": 1,
            "frames_to_tensor": list,
            "stream": SimpleNamespace(push=push),
            "tensor_to_uint8": list,
            "writer": SimpleNamespace(send=lambda frame: output.append(frame[0])),
            "written": 0,
            "padded": 0,
            "started": time.perf_counter(),
            "time": time,
            "print": lambda *args, **kwargs: None,
        }
        run_code(path, [frame_blocks, render_loop], bindings)
        expected_starts = [0, *range(1, length, 4)]
        assert_frame_order(expected, output, f"Matrix-Game-4 output: {length=}, {seed=}")
        assert_frame_order(expected_starts, starts, f"Matrix-Game-4 offsets: {length=}, {seed=}")
    print(f"Matrix-Game-4: {len(lengths)}/{len(lengths)} render boundary cases passed")


def load_pose_code(generate, trajectory):
    import numpy as np
    from scipy.spatial.transform import Rotation
    import torch

    bindings = {"np": np, "R": Rotation, "torch": torch}
    trajectory_functions = {"rot_x", "rot_y", "rot_z", "generate_camera_trajectory_local"}
    run_code(trajectory, (
        node for node in parse(trajectory).body if getattr(node, "name", "") in trajectory_functions
    ), bindings)

    pose_functions = {
        "one_hot_to_one_dimension", "parse_pose_string", "pose_string_to_json", "pose_to_input"
    }
    run_code(generate, (
        node for node in parse(generate).body
        if getattr(node, "name", "") in pose_functions
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mapping" for target in node.targets)
    ), bindings)
    return bindings["pose_string_to_json"], bindings["pose_to_input"]


def check_hy_worldplay(generate, trajectory, before, before_trajectory, seed):
    make_poses, pose_to_input = load_pose_code(generate, trajectory)
    rng = random.Random(seed)
    labels = {"w": 9, "s": 18, "d": 27, "a": 36}
    cases = [[("w", 2), ("s", 1), ("d", 3)]]
    for _ in range(32):
        cases.append([(rng.choice(tuple(labels)), rng.randint(1, 8)) for _ in range(rng.randint(1, 5))])

    for moves in cases:
        command = ", ".join(f"{key}-{count}" for key, count in moves)
        expected = [0, *(labels[key] for key, count in moves for _ in range(count))]
        poses, intrinsics, actions = pose_to_input(command, len(expected))
        if len(poses) != len(expected) or len(intrinsics) != len(expected) or actions.tolist() != expected:
            raise AssertionError(f"HY-WorldPlay: {command!r}, expected {expected}, got {actions.tolist()}")
    print(f"HY-WorldPlay: {len(cases)}/{len(cases)} pose/action ordering cases passed")

    pose_dict = make_poses("w-3, d-2")
    requested = len(pose_dict) - 1
    try:
        pose_to_input(pose_dict, requested)
    except AssertionError as error:
        if "ensure alignment" not in str(error):
            raise
    else:
        raise AssertionError("HY-WorldPlay accepted a mismatched pose count")
    print(f"HY-WorldPlay: rejected {len(pose_dict)} poses for {requested} latent steps")

    if before is not None:
        _, old_pose_to_input = load_pose_code(before, before_trajectory)
        poses, _, _ = old_pose_to_input(pose_dict, requested)
        if len(poses) != requested:
            raise AssertionError("HY-WorldPlay pre-fix control did not reproduce the silent truncation")
        print(f"HY-WorldPlay before #17: silently truncated {len(pose_dict)} poses to {len(poses)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-offline", type=Path, help="Matrix-Game-4 offline.py")
    parser.add_argument("--hy-generate", type=Path, help="HY-WorldPlay hyvideo/generate.py")
    parser.add_argument("--hy-trajectory", type=Path, help="HY-WorldPlay hyvideo/generate_custom_trajectory.py")
    parser.add_argument("--hy-before", type=Path, help="Optional HY-WorldPlay pre-#17 generate.py")
    parser.add_argument("--hy-before-trajectory", type=Path, help="Optional corresponding pre-#17 trajectory file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not any((args.matrix_offline, args.hy_generate, args.hy_trajectory, args.hy_before, args.hy_before_trajectory)):
        hy = UPSTREAM / "hy_worldplay"
        args.matrix_offline = UPSTREAM / "matrix_game_4" / "offline.py"
        args.hy_generate = hy / "generate.py"
        args.hy_trajectory = hy / "generate_custom_trajectory.py"
        args.hy_before = hy / "before_pr17" / "generate.py"
        args.hy_before_trajectory = hy / "before_pr17" / "generate_custom_trajectory.py"
    if bool(args.hy_generate) != bool(args.hy_trajectory):
        parser.error("pass --hy-generate and --hy-trajectory together")
    if bool(args.hy_before) != bool(args.hy_before_trajectory) or args.hy_before and not args.hy_generate:
        parser.error("pass both --hy-before and --hy-before-trajectory alongside the current files")
    if args.matrix_offline:
        check_matrix_game(args.matrix_offline, args.seed)
    if args.hy_generate:
        check_hy_worldplay(args.hy_generate, args.hy_trajectory, args.hy_before, args.hy_before_trajectory, args.seed)


if __name__ == "__main__":
    main()
