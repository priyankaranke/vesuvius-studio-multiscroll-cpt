"""Offline tiled inference on a uint8 [16,H,W] array in verified model depth order.

Default: 128 px windows on a 96 px grid, central 96 px written without overlap (Studio's published tiling).
--stride 48 --blend hann: windows every 48 px, each central 96 px weighted by a pixel-centred 2-D Hann window and the
overlaps averaged by weight. At stride 48 every pixel lies in two windows per axis and the weights sum to one, so no
tile pattern is left. Stride 64 (the tiling of the PHerc0211 submission images) hands over between windows within a
32 px band and leaves faint squares in weak-ink areas.
Pixels with no CT data (all 16 input planes 0, i.e. outside the rendered surface) are written as 0, and windows whose
central 96 px has no CT data are not run at all (their pixels are 0 anyway; --keep-empty runs them).
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from model import DeepResidualUNet


def crop(array, y, x):
    h, w = array.shape[-2:]
    tile = np.asarray(array[:, max(0, y):min(h, y + 128),
                            max(0, x):min(w, x + 128)])
    padding = ((0, 0), (max(0, -y), max(0, y + 128 - h)),
               (max(0, -x), max(0, x + 128 - w)))
    return np.pad(tile, padding, mode="reflect") if any(a or b for a, b in padding) else tile


def load_model(checkpoint, device):
    model = DeepResidualUNet(deeper=False)
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    return model.eval().to(device)


def hann96():
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    return (w[:, None] * w[None, :]).astype(np.float32)


@torch.inference_mode()
def predict(model, volume, output, device="cuda", batch_size=1, amp="bf16", stride=96, blend="none", skip_empty=True):
    if volume.ndim != 3 or volume.shape[0] != 16 or volume.dtype != np.uint8:
        raise ValueError("Expected uint8 [16,H,W], in the verified model depth order")
    h, w = volume.shape[-2:]
    if min(h, w) < 1 or batch_size < 1:
        raise ValueError("Image dimensions and batch size must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(h, w))
    empty = (np.asarray(volume) == 0).all(0)   # no CT data: outside the rendered surface
    if blend == "hann":
        weight = np.zeros((h, w), np.float32); kernel = hann96()
        result[:] = 0
    starts_y = sorted(set(list(range(0, h, stride)) + ([max(h - 96, 0)] if blend == "hann" else [])))
    starts_x = sorted(set(list(range(0, w, stride)) + ([max(w - 96, 0)] if blend == "hann" else [])))
    # A window only writes its central 96 px; if none of them has CT data, its output would be masked to 0 below anyway.
    coords = ((y, x) for y in starts_y for x in starts_x if not (skip_empty and empty[y:y + 96, x:x + 96].all()))
    from itertools import islice
    while batch := list(islice(coords, batch_size)):
        raw = np.stack([crop(volume, y - 16, x - 16) for y, x in batch])
        x = torch.from_numpy(raw[:, None]).to(device=device, dtype=torch.float32)
        x.clamp_(0, 200).div_(255)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") and amp == "bf16" else nullcontext()
        with context:
            logits = model(x)
        probability = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
        for (y, x), p in zip(batch, probability):
            hh, ww = min(96, h - y), min(96, w - x)
            if blend == "hann":
                result[y:y + hh, x:x + ww] += p[:hh, :ww] * kernel[:hh, :ww]
                weight[y:y + hh, x:x + ww] += kernel[:hh, :ww]
            else:
                result[y:y + hh, x:x + ww] = p[:hh, :ww]
    if blend == "hann":
        result[:] = np.where(weight > 0, result / np.maximum(weight, 1e-12), 0).astype(np.float32)
    result[empty] = 0
    result.flush()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local .npy uint8 [16,H,W]")
    parser.add_argument("--output", required=True, help="Output .npy float32 [H,W] probabilities")
    parser.add_argument("--checkpoint", help="Local model.safetensors; otherwise download from --repo")
    parser.add_argument("--repo", default="paranke/vesuvius-studio-multiscroll-cpt-20261023")
    parser.add_argument("--revision", default="main", help="HF commit or revision; pin for reproducibility")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp32"],
                        help="CUDA inference precision. CPU and Apple GPU inference use FP32.")
    parser.add_argument("--stride", type=int, default=96, help="Window step in px; 96 = no overlap (default); with --blend hann, 48 = no tile pattern, 64 = the submission tiling")
    parser.add_argument("--blend", default="none", choices=["none", "hann"], help="hann: Hann-weighted average of overlapping windows")
    parser.add_argument("--keep-empty", action="store_true", help="Also run windows with no CT data (slower)")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("Input and output must be different files")
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new filename")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple GPU (MPS) is unavailable in this PyTorch environment; select --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu or install compatible PyTorch")
    if args.device == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("GPU does not support BF16; use --amp fp32")
    checkpoint = args.checkpoint
    if not checkpoint:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(args.repo, "model.safetensors", revision=args.revision)
    volume = np.load(args.input, mmap_mode="r", allow_pickle=False)
    model = load_model(checkpoint, args.device)
    if args.blend == "none" and args.stride != 96:
        parser.error("--stride other than 96 needs --blend hann")
    predict(model, volume, args.output, args.device, args.batch_size, args.amp, args.stride, args.blend,
            skip_empty=not args.keep_empty)


if __name__ == "__main__":
    main()
