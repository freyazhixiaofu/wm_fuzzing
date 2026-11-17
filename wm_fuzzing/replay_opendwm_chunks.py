"""Capture real OpenDWM CTSD decoder output and replay its stitching on CPU.

Call capture_chunks from a configured batch-size-two OpenDWM session for
[A, B], the repeatability control [A, B], and [A, B_changed]. Reset its random
generator to the same state before each run. Then pass the three resulting
.npz files to this script in that order.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np


def validated(chunks, trims):
    if len(chunks) < 2 or len(chunks) != len(trims):
        raise ValueError("Expected two or more decoded windows with one trim count each.")
    for chunk, trim in zip(chunks, trims):
        if chunk.ndim != 6 or chunk.shape[0] != 2:
            raise ValueError("Each window must have shape (2, time, views, channels, height, width).")
        if (
            chunk.shape[2:] != chunks[0].shape[2:]
            or not isinstance(trim, (int, np.integer))
            or not 0 <= trim < chunk.shape[1]
        ):
            raise ValueError("Windows must use common image dimensions and a valid reference count.")


def stitch(chunks, trims):
    """Return the upstream flat-slice result and the per-sample time-slice result."""
    validated(chunks, trims)
    # CTSD non-diffusion-forcing path, ctsd.py lines 1785–1832 at b0ecc3d.
    _, _, views, channels, height, width = chunks[0].shape
    original = np.concatenate(
        [chunk.reshape(-1, channels, height, width)[2 * trim * views :] for chunk, trim in zip(chunks, trims)]
    ).reshape(2, -1, views, channels, height, width)
    corrected = np.concatenate([chunk[:, trim:] for chunk, trim in zip(chunks, trims)], axis=1)
    return original, corrected


def tensor_array(tensor):
    import torch

    value = tensor.detach().cpu()
    return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()


def capture_chunks(pipeline, latent_shape, batch, output_path):
    """Run the configured sampler once, saving pre-stitch images and trim counts."""
    if latent_shape[0] != 2:
        raise ValueError("Capture needs two samples in the latent batch.")
    if pipeline.common_config.get("frame_prediction_style") == "diffusion_forcing":
        raise ValueError("Select the overlapping CTSD path rather than diffusion forcing.")

    reference = pipeline.inference_config.get("reference_frame_count", 1)
    first_trim = 0 if pipeline.inference_config.get("generate_frames_for_reference", True) else reference
    generate = pipeline.inference_pipeline
    chunks = []

    def record(*args, **kwargs):
        result = generate(*args, **kwargs)
        flat = tensor_array(result["images"])
        chunks.append(flat.reshape(2, -1, latent_shape[2], *flat.shape[1:]).copy())
        return result

    with patch.object(pipeline, "inference_pipeline", new=record):
        result = pipeline.autoregressive_inference_pipeline(latent_shape, batch, "pt")

    trims = [first_trim] + [reference] * (len(chunks) - 1)
    original, _ = stitch(chunks, trims)
    upstream = tensor_array(result["images"]).reshape(original.shape)
    if not np.array_equal(original, upstream):
        raise ValueError("The flat-slice replay disagrees with this pipeline's actual stitched output.")

    np.savez(output_path, *chunks, trims=np.asarray(trims, dtype=np.int64))
    return result


def read_capture(path):
    with np.load(path, allow_pickle=False) as archive:
        trims = archive["trims"].tolist()
        chunks = [archive[f"arr_{index}"] for index in range(len(trims))]
    validated(chunks, trims)
    return chunks, trims


def compare(label, first, second, atol):
    if len(first) != len(second) or any(left.shape != right.shape for left, right in zip(first, second)):
        raise ValueError(f"{label} needs matching image dimensions and window lengths.")
    frames = changed_frames = changed_values = values = 0
    total = maximum = 0.0
    for left, right in zip(first, second):
        diff = np.subtract(left, right, dtype=np.result_type(left.dtype, right.dtype, np.float32))
        np.abs(diff, out=diff)
        if not np.isfinite(diff).all():
            raise ValueError(f"{label} contains non-finite values.")
        different = diff > atol
        frames += diff.shape[0]
        changed_frames += np.count_nonzero(different.reshape(diff.shape[0], -1).any(axis=1))
        values += diff.size
        changed_values += np.count_nonzero(different)
        total += float(diff.sum(dtype=np.float64))
        maximum = max(maximum, float(diff.max()))
    print(f"{label} | {changed_frames}/{frames} frames and {changed_values}/{values} values differ, "
          f"mean absolute difference {total / values:.6g}, max {maximum:.6g}")
    return changed_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("changed_b", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")

    baseline, trims = read_capture(args.baseline)
    repeat, repeat_trims = read_capture(args.repeat)
    changed, changed_trims = read_capture(args.changed_b)
    if repeat_trims != trims or changed_trims != trims:
        parser.error("All runs need identical window boundaries and reference counts")

    repeat_differences = [
        compare(f"Repeatability before stitch, {label}", [c[index] for c in baseline], [c[index] for c in repeat], args.atol)
        for index, label in enumerate(("A", "B"))
    ]
    stable_a = compare("Change B before stitch, A", [c[0] for c in baseline], [c[0] for c in changed], args.atol) == 0
    changed_b = compare("Change B before stitch, B", [c[1] for c in baseline], [c[1] for c in changed], args.atol) != 0
    original, corrected = stitch(baseline, trims)
    original_changed, corrected_changed = stitch(changed, trims)
    bad_a = compare("Change B after original stitch, A", [original[0]], [original_changed[0]], args.atol)
    good_a = compare("Change B after corrected stitch, A", [corrected[0]], [corrected_changed[0]], args.atol)
    for index, label in enumerate(("A", "B")):
        compare(f"Original versus corrected, {label}", [original[index]], [corrected[index]], args.atol)

    if any(repeat_differences) or not stable_a or not changed_b or good_a:
        print("Controls failed. This capture does not isolate the effect to stitching.")
        return 2
    print("Stitch-only cross-sample change detected." if bad_a else "No cross-sample change detected in this capture.")
    return int(bool(bad_a))


if __name__ == "__main__":
    raise SystemExit(main())
