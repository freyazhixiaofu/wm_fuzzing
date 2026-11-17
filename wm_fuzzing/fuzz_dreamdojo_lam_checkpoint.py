"""Measure batch coupling with the real DreamDojo latent-action encoder on CPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import io
import json
from math import prod
import os
from pathlib import Path
import sys
from textwrap import dedent
import time
from types import MethodType, ModuleType, SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, HTTPRedirectHandler, Request, urlopen

from einops import rearrange
import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


SOURCE = Path(__file__).with_name("upstream") / "DreamDojo"
CHECKPOINT_REVISION = "967d74437710976aacad44666eb17d5cf2f3258b"
CHECKPOINT_URL = f"https://huggingface.co/nvidia/DreamDojo/resolve/{CHECKPOINT_REVISION}/LAM_400k.ckpt"
CHECKPOINT_SIZE = 8_518_404_954
CHECKPOINT_SHA256 = "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f"
RANGE_CACHE_BYTES = 1024**2
MAX_RANGE_BYTES = 64 * 1024**2
ENCODER_KEYS = ("lam.action_prompt", "lam.encoder.", "lam.fc.")


def encoder_key(name):
    return name.startswith(ENCODER_KEYS)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class CheckpointRanges(io.RawIOBase):
    """Read only archive metadata and the requested encoder tensor ranges."""

    def __init__(self):
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        try:
            with build_opener(NoRedirect).open(Request(CHECKPOINT_URL, method="HEAD"), timeout=30) as response:
                headers = response.headers
        except HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            headers = error.headers
            error.close()
        size = headers.get("X-Linked-Size", headers.get("Content-Length", 0))
        digest = headers.get("X-Linked-ETag", headers.get("ETag", "")).strip('"')
        if int(size) != CHECKPOINT_SIZE:
            raise RuntimeError("Unexpected official checkpoint size")
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError("Unexpected official checkpoint digest header")

    def fetch(self, start, size):
        end = start + size - 1
        for attempt in range(3):
            request = Request(CHECKPOINT_URL, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urlopen(request, timeout=120) as response:
                    expected_range = f"bytes {start}-{end}/{CHECKPOINT_SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Official checkpoint server did not return the requested byte range")
                    payload = response.read(size + 1)
                if len(payload) == size:
                    return payload
            except HTTPError as error:
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except (OSError, URLError):
                if attempt == 2:
                    raise
            time.sleep(attempt + 1)
        raise RuntimeError("Truncated checkpoint byte range")

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: CHECKPOINT_SIZE}[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Negative checkpoint position")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        remaining = max(0, CHECKPOINT_SIZE - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        if size > MAX_RANGE_BYTES:
            raise RuntimeError("Metadata unexpectedly requested a large checkpoint read")
        relative = self.position - self.cache_start
        if relative < 0 or relative + size > len(self.cache):
            self.cache_start = self.position
            self.cache = self.fetch(self.position, min(remaining, max(size, RANGE_CACHE_BYTES)))
            relative = 0
        self.position += size
        return self.cache[relative : relative + size]

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def grouped_tensor_ranges(tensors):
    while tensors:
        first = tensors[0][0]
        last = 0
        end = first
        for start, size, *_ in tensors:
            if last and (start - end > RANGE_CACHE_BYTES or start + size - first > MAX_RANGE_BYTES):
                break
            end = start + size
            last += 1
        yield first, end - first, tensors[:last]
        tensors = tensors[last:]


def official_encoder_weights():
    with CheckpointRanges() as archive:
        with FakeTensorMode():
            state = torch.load(archive, map_location="cpu", weights_only=True)["state_dict"]
        tensors = []
        for name, tensor in state.items():
            if encoder_key(name):
                storage = tensor.untyped_storage()
                start = storage._checkpoint_offset
                size = storage.nbytes()
                if tensor.storage_offset() or not tensor.is_contiguous() or size != tensor.numel() * tensor.element_size():
                    raise RuntimeError(f"Unsupported layout for {name}")
                tensors.append((start, size, name, tensor.dtype, tuple(tensor.shape)))
        tensors.sort()
        if not tensors:
            raise RuntimeError("Official archive did not contain the expected encoder weights")
        total = sum(size for _, size, _, _, _ in tensors)
        print(f"Loading {len(tensors)} encoder tensors ({total / 1e9:.2f} GB) from the pinned official archive.", flush=True)

        result = {}
        groups = list(grouped_tensor_ranges(tensors))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i in range(0, len(groups), 4):
                requests = groups[i:i + 4]
                futures = [pool.submit(archive.fetch, first, size) for first, size, _ in requests]
                for (first, _, entries), future in zip(requests, futures):
                    chunk = bytearray(future.result())
                    for start, _, name, dtype, shape in entries:
                        value = torch.frombuffer(chunk, dtype=dtype, count=prod(shape), offset=start - first)
                        result[name.removeprefix("lam.")] = value.reshape(shape).clone()
                print(f"Loaded {len(result)}/{len(tensors)} encoder tensors.", flush=True)
        return result


def build_encoder(weights):
    for name, folder in (
        ("external", SOURCE), ("external.lam", SOURCE / "lam"),
        ("external.lam.modules", SOURCE / "lam" / "modules"),
    ):
        package = ModuleType(name)
        package.__path__ = [os.fspath(folder)]
        sys.modules[name] = package
    blocks = import_module("external.lam.modules.blocks")
    lam = import_module("external.lam.modules.lam")
    with torch.device("meta"):
        model = lam.LatentActionModel(3, 1024, 32, 16, 24, 0, 16)
    del model.patch_up, model.action_up, model.decoder
    # Preserve the upstream positional addition while targeting the input device on CPU.
    positions = blocks.PositionalEncoding(1024)
    positions.forward = MethodType(lambda self, x: x + self.pos_enc[:x.shape[2]].to(x.device), positions)
    model.encoder.pos_enc = positions
    model.load_state_dict(weights, strict=True, assign=True)
    # Meta construction avoids allocating a second complete set of encoder weights.
    # The rotary caches are zero-initialized, nonpersistent buffers in this configuration.
    for module in model.modules():
        for name in module._non_persistent_buffers_set:
            buffer = getattr(module, name)
            if buffer is not None and buffer.is_meta:
                setattr(module, name, torch.zeros_like(buffer, device="cpu"))
    return model.eval()


def run_probe(model, frames):
    upstream = dedent((SOURCE / "lam_forward_excerpt.txt").read_text(encoding="utf-8"))
    assembly = compile(upstream, "DreamDojo upstream action assembly", "exec")
    reference = compile(upstream.replace("(t b) d -> b t d", "(b t) d -> b t d"), "Batch-major control", "exec")
    steps = 3
    size = 64

    def synthetic_clip(color, shift):
        result = torch.full((steps, 2, size, size, 3), 0.15)
        for step in range(steps):
            for frame in range(2):
                p = 6 + 3 * step + shift * frame
                result[step, frame, 16:32, p:p + 16, color] = 0.95
        return result.view(steps * 2, size, size, 3)

    if frames:
        with np.load(frames, allow_pickle=False) as extracted:
            a, b, b_changed = (torch.from_numpy(extracted[name]) for name in ("a", "b", "b_changed"))
        for clip in (a, b, b_changed):
            if clip.shape != (steps * 2, size, size, 3) or clip.dtype != torch.float32:
                raise ValueError("Expected six 64x64 RGB float32 frames per sample")
            if not torch.isfinite(clip).all() or clip.min() < 0 or clip.max() > 1:
                raise ValueError("Expected finite RGB pixels in [0, 1]")
        print(f"Input: {frames.name}", flush=True)
    else:
        a, b, b_changed = synthetic_clip(0, 4), synthetic_clip(1, 8), synthetic_clip(2, 14)
        print("Input: deterministic synthetic clips", flush=True)

    def forward(batch, stage, prior=None):
        captured = []

        def encode(inputs):
            output = model(inputs) if prior is None else {"z_rep": prior.reshape(2 * steps, 1, 1, 32)}
            captured.append(output["z_rep"].reshape(2, steps, 32).clone())
            return output

        data = {"lam_video": torch.stack(batch), "action": torch.ones(2, steps, 32)}
        exec(stage, {"torch": torch, "rearrange": rearrange, "self": SimpleNamespace(lam=SimpleNamespace(lam=encode)), "data_batch": data})
        return captured[0], data["action"]

    with torch.inference_mode():
        original_a, coupled_a = forward((a, b), assembly)
        repeated_a, repeated_coupled_a = forward((a, b), assembly)
        changed_a, changed_coupled_a = forward((a, b_changed), assembly)
        _, control_a = forward((a, b), reference, original_a)
        _, changed_control_a = forward((a, b_changed), reference, changed_a)

    def delta(first, second):
        return float((first - second).abs().max())

    outcome = {
        "batch_size": 2,
        "steps": steps,
        "a_before_reshape_repeat_max_abs_delta": delta(original_a[0], repeated_a[0]),
        "a_after_reshape_repeat_max_abs_delta": delta(coupled_a[0], repeated_coupled_a[0]),
        "b_before_reshape_max_abs_delta_when_changed": delta(original_a[1], changed_a[1]),
        "a_before_reshape_max_abs_delta_when_b_changes": delta(original_a[0], changed_a[0]),
        "a_after_reshape_max_abs_delta_when_b_changes": delta(coupled_a[0], changed_coupled_a[0]),
        "a_after_reshape_max_abs_delta_per_step": (coupled_a[0] - changed_coupled_a[0]).abs().amax(dim=1).tolist(),
        "a_after_batch_major_control_max_abs_delta": delta(control_a[0], changed_control_a[0]),
    }
    print(json.dumps(outcome, indent=2), flush=True)
    return int(not (
        outcome["a_before_reshape_repeat_max_abs_delta"] == 0
        and outcome["a_after_reshape_repeat_max_abs_delta"] == 0
        and outcome["b_before_reshape_max_abs_delta_when_changed"] > 0
        and outcome["a_before_reshape_max_abs_delta_when_b_changes"] == 0
        and outcome["a_after_reshape_max_abs_delta_when_b_changes"] > 0
        and outcome["a_after_batch_major_control_max_abs_delta"] == 0
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--checkpoint", type=Path, help="Path to NVIDIA's released LAM_400k.ckpt")
    input_group.add_argument("--official-ranges", action="store_true", help="Load the official encoder tensors over HTTPS")
    parser.add_argument("--frames", type=Path, help="Optional RGB clip arrays a, b, and b_changed in an NPZ file")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        weights = {name.removeprefix("lam."): data for name, data in checkpoint["state_dict"].items() if encoder_key(name)}
    else:
        weights = official_encoder_weights()
    return run_probe(build_encoder(weights), args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
