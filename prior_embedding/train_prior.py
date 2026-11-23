#!/usr/bin/env python3
"""Prior embedding: train a hash-encoding INR on the augmented images.

One network is fitted to all N augmented images at once. Every image j gets its own learnable 2-D latent code z_j;
the network maps (voxel coordinate, z_j) -> intensity of image j. The coordinate hash encoder (encoder.pt) then holds
the anatomy shared by all augmentations, and the reconstruction loads it as its prior.

Network : coordinate hash grid (16 levels x 2 features, 2^19 table, base resolution 16, scale 2)
          + latent hash grid (same configuration, 2-D input) -> fully fused MLP (3 x 64, ReLU) -> 1 output
Training: L1 loss inside the tissue mask, Adam (network lr 1e-3, latent lr 5e-5), StepLR (x0.1 every 200 epochs),
          500 epochs, one augmented image per step in random order, seed 0.

Input : the output of ../augmentation/apply_augmentation.py (augmented_images (N, H, W, D), tissue_mask (H, W, D)).
Output (--out-dir): encoder.pt, latent_encoder.pt, mlp.pt (the prior loaded by the reconstruction),
                    prior_embedding_meta.npz (loss_history, latent_inp, fit_map, target_map) and, with --save-maps,
                    prior_embedding_final_maps.mat (adds the N fitted images fit_maps_all and augmented_targets).

  python train_prior.py
  python train_prior.py --augmented ../augmentation/output/augmented_images.mat --out-dir output --gpu-id 0

Requires a CUDA GPU and tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn, pytorch bindings).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
from torch.optim import lr_scheduler

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None

HERE = Path(__file__).resolve().parent

HASH_GRID = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.0,
    "interpolation": "Linear",
}
MLP = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 3,
}


class PriorINR(nn.Module):
    """f(coordinate, latent) = MLP([hash(coordinate), hash(latent)])."""

    def __init__(self, latent_dim: int = 2) -> None:
        super().__init__()
        if tcnn is None:
            raise ImportError("tinycudann is required (https://github.com/NVlabs/tiny-cuda-nn).")
        self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config=HASH_GRID)
        self.latent_encoder = tcnn.Encoding(n_input_dims=latent_dim, encoding_config=HASH_GRID)
        self.mlp = tcnn.Network(n_input_dims=self.encoder.n_output_dims + self.latent_encoder.n_output_dims,
                                n_output_dims=1, network_config=MLP)

    def forward(self, coords_01: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([self.encoder(coords_01.float()), self.latent_encoder(latent.float())], dim=-1)
        return self.mlp(feats).float()   # tcnn may return fp16


def build_coords(nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
    """Voxel coordinates in [0, 1]^3, (nx * ny * nz, 3), x slowest."""
    axes = [torch.linspace(0.0, 1.0, n, device=device) for n in (nx, ny, nz)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.min()) / torch.clamp(x.max() - x.min(), min=eps)


def load_augmented(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """augmented_images (N, H, W, D) and tissue_mask (H, W, D) from the apply_augmentation.py output."""
    data = loadmat(str(path))
    aug = np.real(np.asarray(data["augmented_images"])).astype(np.float32)
    mask = np.asarray(data["tissue_mask"])
    if aug.ndim != 4 or aug.shape[1:] != mask.shape:
        raise ValueError(f"Expected augmented_images (N, H, W, D) and tissue_mask (H, W, D), got {aug.shape} and {mask.shape}")
    return aug, mask


def train_prior(aug: np.ndarray, mask: np.ndarray, *, epochs: int = 500, lr: float = 1e-3, lr_latent: float = 5e-5,
                lr_step: int = 200, seed: int = 0, device: str = "cuda") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    n_aug, nx, ny, nz = aug.shape
    n_vox = nx * ny * nz
    coords = build_coords(nx, ny, nz, dev)
    background = torch.from_numpy(np.asarray(mask) <= 0).to(dev)
    fg = (~background).float().reshape(-1, 1)

    # Each augmented image rescaled to [0, 1], background set to 0.
    bank = torch.from_numpy(aug).to(dev).masked_fill(background, 0.0)
    bank = torch.stack([normalize01(b) for b in bank]).masked_fill(background, 0.0)

    latent = nn.Parameter(torch.randn((n_aug, 2), dtype=torch.float32, device=dev))
    model = PriorINR().to(dev)
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": lr}, {"params": [latent], "lr": lr_latent}])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
    loss_fn = nn.L1Loss()

    loss_history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for j in rng.permutation(n_aug):
            optimizer.zero_grad(set_to_none=True)
            pred = model(coords, latent[j:j + 1].repeat(n_vox, 1)) * fg
            loss = loss_fn(pred, bank[j].reshape(-1, 1) * fg)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step()
        loss_history.append(epoch_loss / n_aug)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}/{epochs}  loss {loss_history[-1]:.6e}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    with torch.no_grad():   # the fitted image of every latent code; fit_map = their mean
        fit_all = torch.stack([model(coords, latent[j:j + 1].repeat(n_vox, 1)).reshape(nx, ny, nz).masked_fill(background, 0.0)
                               for j in range(n_aug)])
    return {
        "model": model,
        "loss_history": np.asarray(loss_history, dtype=np.float32),
        "latent_inp": latent.detach().cpu().numpy().astype(np.float32),
        "fit_maps_all": fit_all.cpu().numpy().astype(np.float32),
        "fit_map": fit_all.mean(0).masked_fill(background, 0.0).cpu().numpy().astype(np.float32),
        "augmented_targets": bank.cpu().numpy().astype(np.float32),
        "minutes": (time.time() - t0) / 60,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--augmented", type=Path, default=HERE.parent / "augmentation" / "output" / "augmented_images.mat",
                   help="output of apply_augmentation.py")
    p.add_argument("--out-dir", type=Path, default=HERE / "output")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3, help="network learning rate")
    p.add_argument("--lr-latent", type=float, default=5e-5, help="latent-code learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--save-maps", action="store_true", help="also write prior_embedding_final_maps.mat (large)")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (tiny-cuda-nn).")
    torch.cuda.set_device(a.gpu_id)
    aug, mask = load_augmented(a.augmented)
    print(f"{a.augmented}: {aug.shape[0]} augmented images {aug.shape[1:]}, {int((mask > 0).sum())} tissue voxels", flush=True)

    res = train_prior(aug, mask, epochs=a.epochs, lr=a.lr, lr_latent=a.lr_latent, seed=a.seed, device=f"cuda:{a.gpu_id}")

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    model = res["model"]
    torch.save(model.encoder.state_dict(), out / "encoder.pt")
    torch.save(model.latent_encoder.state_dict(), out / "latent_encoder.pt")
    torch.save(model.mlp.state_dict(), out / "mlp.pt")
    target_map = res["augmented_targets"][0]   # first augmented image, normalized (kept for reference)
    np.savez_compressed(out / "prior_embedding_meta.npz", loss_history=res["loss_history"], latent_inp=res["latent_inp"],
                        fit_map=res["fit_map"], target_map=target_map, minutes=res["minutes"])
    if a.save_maps:
        savemat(str(out / "prior_embedding_final_maps.mat"),
                {k: res[k] for k in ("fit_map", "fit_maps_all", "augmented_targets", "latent_inp", "loss_history")}
                | {"target_map": target_map})
    print(f"saved prior to {out} ({res['minutes']:.1f} min, final loss {res['loss_history'][-1]:.6e})")


if __name__ == "__main__":
    main()
