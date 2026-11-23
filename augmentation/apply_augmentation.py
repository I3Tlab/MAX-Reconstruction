#!/usr/bin/env python3
"""Apply the selected contrast augmentations to a reference image.

Each entry of the augmentation list (selected_augmentations.json) is a chain of intensity transformations with fixed,
recorded parameters (invert, affine, gamma, log, sigmoid, piecewise, taylor, taylor_centered). Every chain is applied
to the reference image (e.g. the MPRAGE registered to the FLAIR grid); the result is restricted to the tissue mask and
rescaled to [0, 1] inside it. The augmented images are the training targets of the prior.

Output (.mat): augmented_images (N, H, W, D), tissue_mask (H, W, D), original_image (H, W, D), plus descriptors,
chain_lengths, chain_params, selection_ranks, source_indices.

  python apply_augmentation.py
  python apply_augmentation.py --reference data/Mprage2Flair.mat --reference-key Mprage2Flair \
      --mask data/Flair_tissue_mask.mat --mask-key tissue_mask \
      --augmentations data/selected_augmentations.json --out output/augmented_images.mat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def load_mat_array(path: Path, key: str) -> np.ndarray:
    """Read one array from a MATLAB file (v5 via scipy, v7.3 via h5py)."""
    try:
        data = loadmat(str(path))
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}; keys = {[k for k in data if not k.startswith('__')]}")
        return np.asarray(data[key])
    except NotImplementedError:   # MATLAB v7.3 = HDF5, stored column-major
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {path}; keys = {list(f)}")
            return np.asarray(f[key][()]).T


def prepare_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Tissue mask restricted to non-zero voxels of the reference image."""
    return ((np.asarray(mask, dtype=np.float32) > 0.5) & (image != 0)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Intensity transformations
# --------------------------------------------------------------------------------------
def normalize01(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float32)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    return ((x - x_min) / max(x_max - x_min, eps)).astype(np.float32), x_min, x_max


def normalize01_in_mask(x: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32) > 0.5
    if not np.any(m):
        return out
    vals = np.asarray(x, dtype=np.float32)[m]
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    out[m] = 0.0 if vmax <= vmin + eps else ((vals - vmin) / (vmax - vmin)).astype(np.float32)
    return out


def apply_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    m = np.asarray(mask) > 0.5
    vals = np.asarray(x, dtype=np.float32)[m]
    out[m] = np.where(np.isfinite(vals), vals, 0.0)
    return out


def polynomial5(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c[0] + c[1] * x + c[2] * (x * x) + c[3] * (x * x * x) + c[4] * (x * x * x * x) + c[5] * (x * x * x * x * x)


def apply_one_transform(x: np.ndarray, params: dict, clip_range: tuple[float, float] | None) -> np.ndarray:
    """One recorded transformation. 'invert' and 'affine' act on the raw intensities, the rest on x rescaled to [0, 1]."""
    op = str(params.get("op", params.get("mode", ""))).strip().lower()
    x = np.asarray(x, dtype=np.float32)

    if op == "invert":   # flip the contrast within the original intensity range
        x01, x_min, x_max = normalize01(x)
        return ((1.0 - x01) * (x_max - x_min) + x_min).astype(np.float32)
    if op == "affine":
        y = float(params["scale"]) * x + float(params["offset"])
        return (y if clip_range is None else np.clip(y, *clip_range)).astype(np.float32)

    x01 = np.clip(normalize01(x)[0], 0.0, 1.0)
    if op == "gamma":
        return np.power(x01, float(params["gamma"])).astype(np.float32)
    if op == "log":
        alpha = float(params["alpha"])
        return (np.log1p(alpha * x01) / np.log1p(alpha)).astype(np.float32)
    if op == "sigmoid":
        y01 = 1.0 / (1.0 + np.exp(-float(params["gain"]) * (x01 - float(params["center"]))))
        return normalize01(y01)[0]
    if op == "piecewise":
        xk, yk = np.asarray(params["xk"], dtype=np.float32), np.asarray(params["yk"], dtype=np.float32)
        return np.interp(x01.reshape(-1), xk, yk).reshape(x01.shape).astype(np.float32)
    if op == "taylor":
        return polynomial5(x01, np.asarray(params["coeffs"], dtype=np.float32)).astype(np.float32)
    if op == "taylor_centered":
        return polynomial5(x01 - np.float32(0.5), np.asarray(params["coeffs"], dtype=np.float32)).astype(np.float32)
    raise ValueError(f"Unsupported transformation: {op}")


def apply_chain(image: np.ndarray, chain: list[dict], mask: np.ndarray, clip_range: tuple[float, float] | None) -> np.ndarray:
    """Apply a chain of transformations in order, then keep the tissue mask and rescale to [0, 1] inside it."""
    x = np.asarray(image, dtype=np.float32).copy()
    for params in chain:
        x = apply_one_transform(x, params, clip_range)
    return normalize01_in_mask(apply_mask(x, mask), mask)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, default=HERE / "data" / "Mprage2Flair.mat", help="reference image (.mat)")
    p.add_argument("--reference-key", default="Mprage2Flair")
    p.add_argument("--mask", type=Path, default=HERE / "data" / "Flair_tissue_mask.mat", help="tissue mask (.mat)")
    p.add_argument("--mask-key", default="tissue_mask")
    p.add_argument("--augmentations", type=Path, default=HERE / "data" / "selected_augmentations.json")
    p.add_argument("--out", type=Path, default=HERE / "output" / "augmented_images.mat")
    a = p.parse_args()

    spec = json.loads(a.augmentations.read_text())
    selected = spec["selected_augmentations"]
    clip = spec.get("clip_range")
    clip_range = None if clip is None else (float(clip[0]), float(clip[1]))

    image = np.real(load_mat_array(a.reference, a.reference_key)).astype(np.float32)
    mask = prepare_mask(load_mat_array(a.mask, a.mask_key), image)
    if image.shape != mask.shape:
        raise ValueError(f"Reference shape {image.shape} and mask shape {mask.shape} differ")
    print(f"reference {a.reference.name} {image.shape}, mask {int(mask.sum())} voxels, {len(selected)} augmentations")

    augmented = np.stack([apply_chain(image, item["params"], mask, clip_range) for item in selected]).astype(np.float32)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(a.out), {
        "original_image": image.astype(np.complex64),   # complex64 as read by the prior-training code
        "tissue_mask": mask,
        "augmented_images": augmented,
        "descriptors": np.asarray([str(item.get("descriptor", "")) for item in selected], dtype=object),
        "chain_lengths": np.asarray([int(item.get("chain_length", len(item["params"]))) for item in selected], dtype=np.int32),
        "chain_params": np.asarray([json.dumps(item["params"], sort_keys=True) for item in selected], dtype=object),
        "selection_ranks": np.asarray([int(item.get("selection_rank", i + 1)) for i, item in enumerate(selected)], dtype=np.int32),
        "source_indices": np.asarray([int(item.get("source_index", -1)) for item in selected], dtype=np.int32),
    })
    print(f"saved {a.out}  augmented_images {augmented.shape}")


if __name__ == "__main__":
    main()
