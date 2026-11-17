"""Run OpenDWM's CTSD stitching code with numbered frame tensors."""

import ast
from types import SimpleNamespace

import torch

from wm_fuzzing.sources import UPSTREAM


def _load_function(source, name, bindings):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    member = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = dict(bindings)
    exec(compile(ast.Module(body=[member], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


def load_rollout(checkout=None):
    """Load the upstream stitch and sequence slicer without importing OpenDWM."""
    snapshot = UPSTREAM / "opendwm"
    if checkout is None:
        clip_source = snapshot / "take_sequence_clip.py"
        rollout_source = snapshot / "ctsd_autoregressive.py"
    else:
        clip_source = checkout / "src/dwm/functional.py"
        rollout_source = checkout / "src/dwm/pipelines/ctsd.py"
    clip = _load_function(clip_source, "take_sequence_clip", {"torch": torch})
    dwm = SimpleNamespace(functional=SimpleNamespace(take_sequence_clip=clip))
    return _load_function(
        rollout_source,
        "autoregressive_inference_pipeline",
        {"torch": torch, "dwm": dwm},
    )


def run_tagged_rollout(rollout, ids, chunk_size, overlap):
    def generate(latent_shape, window, output_type, condition, reference_frame_count):
        tags = window["pts"]
        assert tags.shape == latent_shape
        if condition is not None:
            torch.testing.assert_close(condition, tags[:, :reference_frame_count])
        # The real decoder returns its image batch flattened in sample/time/view order.
        return {"latents": tags, "images": tags.flatten(0, 2)}

    pipeline = SimpleNamespace(
        common_config={},
        inference_config={"sequence_length_per_iteration": chunk_size, "reference_frame_count": overlap},
        inference_pipeline=generate,
        get_latent_sequence_length=int,
    )
    shape = (ids.shape[0], chunk_size, *ids.shape[2:])
    return rollout(pipeline, shape, {"pts": ids}, "pt")["images"]
