#!/usr/bin/env python3
"""MAX reconstruction of undersampled multi-coil 2D k-space with the pre-trained prior and the reference image.

Stage 1 (INR fit): a hash-encoding INR whose coordinate encoder is the frozen prior (../prior_embedding/output/encoder.pt)
    is fitted to the measured k-space: L1 data consistency, 500 steps, Adam lr 1e-3 (latent codes 1e-4), lr x0.1 after
    250 steps. Its output is x_inr.
Stage 2 (reference-guided unrolled refinement): 8 unrolled blocks, each
    x_cnn = CNN([x, reference]) (ResNet, 15 blocks x 64 channels, residual on x),
    x     = argmin ||A x - y||^2 + lambda ||x - x_cnn||^2   (10 CG iterations, lambda learnable, init 0.05),
    started from x = x_inr. The reference (registered to the target grid, tissue-masked, scaled to max 1 per slice) is an
    extra CNN input channel in every block. Trained self-supervised on the measured k-space (ZS-SSL: 5 % of the sampled
    points held out for validation, the rest re-split every epoch into 60 % data consistency / 40 % training loss),
    Adam lr 2e-4, up to 2000 epochs, early stop after 25 epochs without validation improvement; the best epoch is kept.

Data (MATLAB files, image axes x, y, slice; coils last):
  --kspace     k-space (nx, ny, nz, ncoil), complex; points outside the sampling mask are ignored
  --csm        coil sensitivities (nx, ny, nz, ncoil), complex
  --tissue-mask (nx, ny, nz)       --reference registered reference image (nx, ny, nz)
  --sampling-mask (nx, ny) 1-D phase-encoding mask, or --acc R to draw a Gaussian 1-D mask (12 ACS lines, seed 3598)
  --prior      folder with encoder.pt and latent_encoder.pt (output of ../prior_embedding/train_prior.py)
Output (--out, .mat): pred_image (MAX), x_inr (stage 1), zf_image, SamMask, solve_scale, loss_history, modl_val_history,
  best_epoch, dc_lambda. If the k-space is fully sampled, PSNR / SSIM of x_inr and pred_image against the fully sampled
  coil-combined image are printed and saved (per-slice least-squares intensity matching inside the tissue mask).

  python reconstruct.py
  python reconstruct.py --acc 12 --out output/recon_R12.mat --gpu-id 1

Requires a CUDA GPU, PyTorch, tiny-cuda-nn, numpy, scipy, h5py (scikit-image for SSIM).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None

HERE = Path(__file__).resolve().parent


@dataclass
class Config:
    # stage 1: INR fit
    steps: int = 500
    lr: float = 1e-3
    latent_lr: float = 1e-4
    # stage 2: reference-guided unrolled refinement
    modl_steps: int = 2000
    modl_lr: float = 2e-4
    n_unrolls: int = 8
    dc_lambda_init: float = 0.05
    cg_max_iter: int = 10
    cg_tol: float = 1e-10
    slice_batch_size: int = 10
    val_ratio: float = 0.05
    train_loss_ratio: float = 0.4
    patience: int = 25
    split_seed: int = 0
    seed: int = 3598
    log_every: int = 50


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def load_mat(path: Path, key: str) -> np.ndarray:
    """One array from a MATLAB file: v5 via scipy, v7.3 (HDF5, column-major, complex as real/imag) via h5py."""
    try:
        return np.asarray(sio.loadmat(str(path), variable_names=[key])[key])
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            arr = f[key][()]
        if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
            arr = arr["real"] + 1j * arr["imag"]
        return np.ascontiguousarray(np.transpose(arr, tuple(range(arr.ndim))[::-1]))


def gaussian_mask_1d(shape: tuple[int, int], acc: int, acs: int = 12, seed: int = 3598) -> np.ndarray:
    """1-D Gaussian random phase-encoding mask along axis 1: acs central lines + Gaussian-distributed lines up to ny // acc."""
    rng = np.random.default_rng(seed)
    nx, ny = int(shape[0]), int(shape[1])
    line = np.zeros((ny,), dtype=np.float32)
    x0 = ny // 2 - acs // 2
    line[x0:x0 + acs] = 1.0
    count, target = int(line.sum()), max(1, ny // acc)
    while count < target:
        i = int(rng.normal(loc=ny / 2.0, scale=max(ny / 5.0, 1e-6)))
        if 0 <= i < ny and line[i] == 0:
            line[i] = 1.0
            count += 1
    return np.repeat(line[None, :], nx, axis=0)


# --------------------------------------------------------------------------------------
# MRI operators (centred FFT over the two in-plane axes)
# --------------------------------------------------------------------------------------
def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1))


def A_forward(img: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """Image (nx, ny, nz) -> multi-coil k-space (nx, ny, nz, ncoil)."""
    return fft2c(img.unsqueeze(-1) * sens)


def A_adjoint(kspace: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    return (ifft2c(kspace) * torch.conj(sens)).sum(dim=3)


def cg_solve(apply_normal: Callable[[torch.Tensor], torch.Tensor], rhs: torch.Tensor, *, max_iter: int, tol: float) -> torch.Tensor:
    def inner(a, b):
        return torch.sum(torch.conj(a) * b).real

    x = torch.zeros_like(rhs)
    r = rhs - apply_normal(x)
    p = r.clone()
    rtr = inner(r, r)
    for _ in range(max_iter):
        if float(rtr.detach().item()) < tol:
            break
        ap = apply_normal(p)
        alpha = rtr / inner(p, ap).clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * ap
        rtr_new = inner(r, r)
        p = r + rtr_new / (rtr + 1e-12) * p
        rtr = rtr_new
    return x


def complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """(nx, ny, nz) complex -> (nz, 2, nx, ny) real."""
    return torch.stack([x.real, x.imag], dim=0).permute(3, 0, 1, 2).contiguous().float()


def twoch_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1]).permute(1, 2, 0).contiguous()


def mix_l1_l2(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """0.5 * relative L2 + 0.5 * relative L1."""
    return 0.5 * torch.norm(yhat - y) / (torch.norm(y) + eps) + 0.5 * torch.norm(yhat - y, p=1) / (torch.norm(y, p=1) + eps)


# --------------------------------------------------------------------------------------
# Stage 1: INR with the prior encoder
# --------------------------------------------------------------------------------------
HASH_GRID = {"otype": "Grid", "type": "Hash", "n_levels": 16, "n_features_per_level": 2, "log2_hashmap_size": 19,
             "base_resolution": 16, "per_level_scale": 2.0, "interpolation": "Linear"}
MLP = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 64, "n_hidden_layers": 3}


class ReconINR(nn.Module):
    """Complex image = MLP([prior(coordinate), w1 * latent1 + w2 * latent2]) x unit phase(coordinate).

    shared_encoder is the prior's coordinate encoder (frozen); both latent encoders start from the prior's latent encoder;
    the two latent codes vary along x and are mixed by coordinate-dependent softmax weights.
    """

    def __init__(self) -> None:
        super().__init__()
        if tcnn is None:
            raise ImportError("tinycudann is required (https://github.com/NVlabs/tiny-cuda-nn).")
        self.shared_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=HASH_GRID)
        self.latent_encoder = tcnn.Encoding(n_input_dims=2, encoding_config=HASH_GRID)
        self.latent_encoder2 = tcnn.Encoding(n_input_dims=2, encoding_config=HASH_GRID)
        self.weight_net = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=2, network_config=MLP,
                                                        encoding_config={"otype": "Frequency", "n_frequencies": 4})
        self.mlp = tcnn.Network(n_input_dims=self.shared_encoder.n_output_dims + self.latent_encoder.n_output_dims,
                                n_output_dims=2, network_config=MLP)
        self.phase_net = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=2, network_config=MLP,
                                                       encoding_config={"otype": "Frequency", "n_frequencies": 6})

    def forward(self, coords_01: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        enc = self.shared_encoder(coords_01.float())
        w = torch.softmax(self.weight_net(coords_01.float()), dim=1)
        enc_latent = self.latent_encoder(latent[0]) * w[:, 0:1] + self.latent_encoder2(latent[1]) * w[:, 1:2]
        mag = self.mlp(torch.cat([enc, enc_latent], dim=-1)).float()   # tcnn may return fp16
        out = torch.complex(mag[:, :1], mag[:, 1:])
        ph = self.phase_net(coords_01.float()).float()
        ph = torch.complex(ph[:, :1], ph[:, 1:])
        return out * (ph / torch.sqrt(ph.conj() * ph + 1e-8))

    def load_prior(self, prior_dir: Path) -> None:
        latent_state = torch.load(prior_dir / "latent_encoder.pt", map_location="cpu", weights_only=True)
        self.shared_encoder.load_state_dict(torch.load(prior_dir / "encoder.pt", map_location="cpu", weights_only=True))
        self.latent_encoder.load_state_dict(latent_state)
        self.latent_encoder2.load_state_dict(latent_state)
        for p in self.shared_encoder.parameters():
            p.requires_grad = False


def build_coords(nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
    axes = [torch.linspace(0.0, 1.0, n, device=device) for n in (nx, ny, nz)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def latent_to_voxels(latent: torch.Tensor, nx: int, ny: int, nz: int) -> torch.Tensor:
    """(2, nx, 1, 1, 2) latent grid -> (2, nx * ny * nz, 2), trilinear upsampling."""
    up = F.interpolate(latent.permute(0, 4, 1, 2, 3), size=(nx, ny, nz), mode="trilinear", align_corners=False)
    return up.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], nx * ny * nz, latent.shape[-1])


def fit_inr(kspace: torch.Tensor, sens: torch.Tensor, mask: torch.Tensor, tissue: torch.Tensor, prior_dir: Path,
            cfg: Config) -> tuple[torch.Tensor, np.ndarray]:
    nx, ny, nz, _ = kspace.shape
    dev = kspace.device
    coords = build_coords(nx, ny, nz, dev)
    latent = nn.Parameter(torch.randn((2, nx, 1, 1, 2), dtype=torch.float32, device=dev))
    model = ReconINR().to(dev)
    model.load_prior(prior_dir)
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": cfg.lr}, {"params": [latent], "lr": cfg.latent_lr}])
    sched = lr_scheduler.StepLR(opt, step_size=max(1, cfg.steps // 2), gamma=0.1)
    l1 = nn.L1Loss()

    def image():
        return model(coords, latent_to_voxels(latent, nx, ny, nz)).reshape(nx, ny, nz) * tissue

    hist = []
    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        pk = A_forward(image(), sens)
        loss = l1(torch.view_as_real(pk[mask]).float(), torch.view_as_real(kspace[mask]).float())
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.item()))
        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            print(f"[stage 1] step {step:4d}/{cfg.steps}  k-space L1 {hist[-1]:.6e}", flush=True)
    with torch.no_grad():
        return image(), np.asarray(hist, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Stage 2: reference-guided unrolled network
# --------------------------------------------------------------------------------------
class ResBlock2D(nn.Module):
    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.act(self.conv1(x))))


class GuidedDenoiser(nn.Module):
    """ResNet on [x / rms(x), reference / max(reference)] (3 channels) -> residual added to x (2 channels)."""

    def __init__(self, feat_ch: int = 64, n_blocks: int = 15) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, feat_ch, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ResBlock2D(feat_ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(feat_ch, 2, 3, padding=1, bias=False)
        nn.init.zeros_(self.tail.weight)   # starts as the identity

    def forward(self, x_2ch: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x_2ch * x_2ch, dim=(1, 2, 3), keepdim=True) + 1e-8)
        g = guide / (torch.amax(guide, dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x_2ch + self.tail(self.blocks(self.head(torch.cat([x_2ch / norm, g], dim=1)))) * norm


class GuidedUnrolledNet(nn.Module):
    def __init__(self, denoiser: nn.Module, n_unrolls: int, dc_lambda_init: float, cg_max_iter: int, cg_tol: float) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.n_unrolls, self.cg_max_iter, self.cg_tol = int(n_unrolls), int(cg_max_iter), float(cg_tol)
        self.dc_lambda = nn.Parameter(torch.tensor([float(dc_lambda_init)]))

    def forward(self, x_init, sens, kspace, mask, tissue, guide):
        """All inputs for a batch of slices: images (nx, ny, b), coil arrays (nx, ny, b, ncoil), guide (nx, ny, b)."""
        x = x_init
        atb = A_adjoint(kspace * mask, sens)
        lam = self.dc_lambda[0]
        g = guide.permute(2, 0, 1).unsqueeze(1).float()

        def normal_op(v):
            return A_adjoint(A_forward(v, sens) * mask, sens) + lam * v

        for _ in range(self.n_unrolls):
            x_cnn = twoch_to_complex(self.denoiser(complex_to_2ch(x), g)) * tissue
            x = cg_solve(normal_op, atb + lam * x_cnn, max_iter=self.cg_max_iter, tol=self.cg_tol) * tissue
        return x


def _split_rows(mask_t: torch.Tensor, exclude: torch.Tensor | None, fractions: tuple[float, ...],
                rng: np.random.Generator) -> list[torch.Tensor]:
    """Per slice, split the sampled points (minus `exclude`) into len(fractions) + 1 disjoint masks; the 4 x 4 centre of
    k-space always goes to the last (data-consistency) mask. Returns masks shaped like mask_t."""
    nx, ny, nz, nc = mask_t.shape
    outs = [torch.zeros_like(mask_t, dtype=torch.bool) for _ in range(len(fractions) + 1)]
    cx, cy = nx // 2, ny // 2
    for z in range(nz):
        base = mask_t[:, :, z, 0] > 0
        if exclude is not None:
            base = base & ~(exclude[:, :, z, 0] > 0)
        pts = torch.nonzero(base, as_tuple=False)
        if pts.numel() == 0:
            continue
        centre = (pts[:, 0] >= cx - 2) & (pts[:, 0] < cx + 2) & (pts[:, 1] >= cy - 2) & (pts[:, 1] < cy + 2)
        protected, cand = pts[centre], pts[~centre]
        cand = cand[torch.as_tensor(rng.permutation(cand.shape[0]), device=mask_t.device, dtype=torch.long)]
        n = cand.shape[0]
        parts, start = [], 0
        for i, frac in enumerate(fractions):
            k = int(round(frac * n))
            if len(fractions) == 2 and i == 0:   # validation split: at least 1, leave >= 2
                k = max(1, min(n - 2, k)) if n >= 3 else max(0, min(n - 1, k))
            else:
                rem = n - start
                k = 0 if rem <= 1 else max(1, min(rem - 1, k))
            parts.append(cand[start:start + k])
            start += k
        parts.append(torch.cat([cand[start:], protected], dim=0))
        for o, p in zip(outs, parts):
            if p.numel():
                o[p[:, 0], p[:, 1], z, :] = True
    return outs


def refine(x_inr, kspace, sens, mask, tissue, guide, cfg: Config):
    dev = kspace.device
    nz = x_inr.shape[2]
    net = GuidedUnrolledNet(GuidedDenoiser(), cfg.n_unrolls, cfg.dc_lambda_init, cfg.cg_max_iter, cfg.cg_tol).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.modl_lr)

    split_rng = np.random.default_rng(cfg.split_seed)
    val_seed = int(split_rng.integers(0, 2**31 - 1))
    epoch_seeds = split_rng.integers(0, 2**31 - 1, size=cfg.modl_steps, dtype=np.int64)
    val_mask = _split_rows(mask, None, (cfg.val_ratio, cfg.train_loss_ratio), np.random.default_rng(val_seed))[0]

    def run(sl, dc_mask):
        return net(x_inr[:, :, sl], sens[:, :, sl], kspace[:, :, sl], dc_mask[:, :, sl], tissue[:, :, sl], guide[:, :, sl])

    def kloss(pred, sl, loss_mask):
        m = loss_mask[:, :, sl]
        return mix_l1_l2(torch.view_as_real(A_forward(pred, sens[:, :, sl])[m]).float(),
                         torch.view_as_real(kspace[:, :, sl][m]).float())

    best, best_ep, bad, best_state, val_hist = float("inf"), -1, 0, copy.deepcopy(net.state_dict()), []
    for ep in range(1, cfg.modl_steps + 1):
        train_mask, dc_mask = _split_rows(mask, val_mask, (cfg.train_loss_ratio,), np.random.default_rng(int(epoch_seeds[ep - 1])))
        perm = torch.randperm(nz, device=dev)
        for s0 in range(0, nz, cfg.slice_batch_size):
            sl = torch.sort(perm[s0:s0 + cfg.slice_batch_size]).values
            opt.zero_grad(set_to_none=True)
            loss = kloss(run(sl, dc_mask), sl, train_mask)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(np.mean([kloss(run(sl, dc_mask), sl, val_mask).item()
                               for sl in (torch.arange(s0, min(s0 + cfg.slice_batch_size, nz), device=dev)
                                          for s0 in range(0, nz, cfg.slice_batch_size))]))
        val_hist.append(v)
        if v < best:
            best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if ep == 1 or ep % 25 == 0:
            print(f"[stage 2] epoch {ep:4d}  val {v:.5e}  best {best:.5e} @ {best_ep}  lambda {net.dc_lambda.item():.4f}", flush=True)
        if bad >= cfg.patience:
            print(f"[stage 2] early stop at epoch {ep}, best epoch {best_ep}", flush=True)
            break
    net.load_state_dict(best_state)
    with torch.no_grad():   # final pass with all measured k-space in the data consistency
        pred = torch.cat([run(torch.arange(s0, min(s0 + cfg.slice_batch_size, nz), device=dev), mask)
                          for s0 in range(0, nz, cfg.slice_batch_size)], dim=2)
    return pred, np.asarray(val_hist, dtype=np.float32), best_ep, float(net.dc_lambda.item())


# --------------------------------------------------------------------------------------
# Evaluation (only possible when the input k-space is fully sampled)
# --------------------------------------------------------------------------------------
def evaluate(recon: np.ndarray, ref: np.ndarray, tissue: np.ndarray) -> dict[str, float]:
    """Per-slice least-squares intensity matching inside the tissue mask, then volume PSNR / 3-D SSIM with data range =
    max of the reference in the mask."""
    m = tissue > 0
    ref_eval = ref * m
    rec_eval = np.zeros_like(ref_eval)
    rec = np.abs(recon).astype(np.float32)
    for z in range(ref.shape[2]):
        mz = m[:, :, z]
        if mz.any():
            a = float(np.sum(rec[:, :, z][mz] * ref[:, :, z][mz]) / (np.sum(rec[:, :, z][mz] ** 2) + 1e-12))
            rec_eval[:, :, z] = a * rec[:, :, z] * mz
    data_range = float(ref[m].max())
    mse = float(np.mean((rec_eval.astype(np.float64) - ref_eval) ** 2))
    out = {"psnr": 20 * math.log10(data_range) - 10 * math.log10(mse)}
    try:
        from skimage.metrics import structural_similarity
        out["ssim"] = float(structural_similarity(rec_eval.astype(np.float64), ref_eval.astype(np.float64),
                                                  data_range=data_range, win_size=7))
    except ImportError:
        pass
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = HERE.parent / "MAX_data" / "brain_FLAIR"
    p.add_argument("--kspace", type=Path, default=d / "Flair_kspace.mat")
    p.add_argument("--kspace-key", default="Flair_kspace")
    p.add_argument("--csm", type=Path, default=d / "Flair_tstCsm.mat")
    p.add_argument("--csm-key", default="Flair_tstCsm")
    p.add_argument("--tissue-mask", type=Path, default=d / "Flair_tissue_mask.mat")
    p.add_argument("--tissue-mask-key", default="tissue_mask")
    p.add_argument("--reference", type=Path, default=d / "Mprage2Flair.mat")
    p.add_argument("--reference-key", default="Mprage2Flair")
    p.add_argument("--sampling-mask", type=Path, default=d / "Flair_SamMask.mat")
    p.add_argument("--sampling-mask-key", default="SamMask")
    p.add_argument("--acc", type=int, default=None, help="draw a Gaussian 1-D mask with this acceleration instead")
    p.add_argument("--prior", type=Path, default=HERE.parent / "prior_embedding" / "output")
    p.add_argument("--out", type=Path, default=HERE / "output" / "recon.mat")
    p.add_argument("--modl-lr", type=float, default=Config.modl_lr, help="stage-2 learning rate (paper: FLAIR 2e-4, T2w 5e-4)")
    p.add_argument("--gpu-id", type=int, default=0)
    a = p.parse_args()

    cfg = Config(modl_lr=a.modl_lr)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (tiny-cuda-nn).")
    torch.cuda.set_device(a.gpu_id)
    dev = torch.device(f"cuda:{a.gpu_id}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    kspace_np = load_mat(a.kspace, a.kspace_key).astype(np.complex64)
    csm_np = load_mat(a.csm, a.csm_key).astype(np.complex64)
    tissue_np = (load_mat(a.tissue_mask, a.tissue_mask_key) > 0).astype(np.float32)
    ref_np = np.abs(load_mat(a.reference, a.reference_key)).astype(np.float32)
    nx, ny, nz, nc = kspace_np.shape
    if csm_np.shape != kspace_np.shape or tissue_np.shape != (nx, ny, nz) or ref_np.shape != (nx, ny, nz):
        raise ValueError(f"shapes: kspace {kspace_np.shape}, csm {csm_np.shape}, tissue {tissue_np.shape}, reference {ref_np.shape}")
    sam = gaussian_mask_1d((nx, ny), a.acc) if a.acc else load_mat(a.sampling_mask, a.sampling_mask_key).astype(np.float32)
    print(f"k-space {kspace_np.shape}, R = {sam.size / sam.sum():.2f}, prior {a.prior}", flush=True)

    kspace = torch.from_numpy(kspace_np).to(dev)
    sens = torch.from_numpy(csm_np).to(dev)
    tissue = torch.from_numpy(tissue_np).to(dev)
    mask = torch.from_numpy(sam > 0).to(dev)[:, :, None, None].expand(nx, ny, nz, nc)
    k_meas = kspace * mask
    zf = torch.sqrt((ifft2c(k_meas).abs() ** 2).sum(dim=3))
    scale = zf.max().clamp_min(1e-8)   # the solve runs on k-space scaled to a zero-filled RSS maximum of 1
    k_meas = k_meas / scale

    t0 = time.time()
    x_inr, loss_hist = fit_inr(k_meas, sens, mask, tissue, a.prior, cfg)
    t1 = time.time()
    guide = torch.from_numpy(ref_np).to(dev) * tissue
    pred, val_hist, best_ep, dc_lambda = refine(x_inr, k_meas, sens, mask, tissue, guide, cfg)
    t2 = time.time()
    print(f"stage 1 {(t1 - t0) / 60:.1f} min, stage 2 {(t2 - t1) / 60:.1f} min (best epoch {best_ep})", flush=True)

    x_inr_np = (x_inr * scale).cpu().numpy().astype(np.complex64)
    pred_np = (pred * scale).cpu().numpy().astype(np.complex64)
    out = {"pred_image": pred_np, "x_inr": x_inr_np, "zf_image": zf.cpu().numpy().astype(np.float32), "SamMask": sam,
           "solve_scale": float(scale), "loss_history": loss_hist, "modl_val_history": val_hist, "best_epoch": best_ep,
           "dc_lambda": dc_lambda, "minutes": (t2 - t0) / 60, "config": json.dumps(asdict(cfg))}
    if np.count_nonzero(np.abs(kspace_np[:, :, nz // 2, 0]).sum(0)) == ny:   # all phase-encoding lines present
        ref_full = np.abs((np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_np, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
                           * np.conj(csm_np)).sum(-1)).astype(np.float32)
        metrics = {"x_inr": evaluate(x_inr_np, ref_full, tissue_np), "pred_image": evaluate(pred_np, ref_full, tissue_np)}
        print("metrics vs fully sampled:", json.dumps(metrics), flush=True)
        out["metrics"] = json.dumps(metrics)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(a.out), out, do_compression=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
