#!/usr/bin/env python3
"""MAX reconstruction of undersampled knee k-space (multi-coil 2D), with the reference image as an extra CNN input in the
unrolled stage.

Stage 1 (INR fit, whole volume): hash-encoding INR whose coordinate encoder is the frozen prior
    (../prior_embedding/output_Knee/encoder.pt) and whose 4 latent codes (each varying along x) are mixed by
    coordinate-dependent softmax weights; L1 k-space data consistency, 1000 steps, Adam lr 1e-3 (latent codes 1e-4),
    lr x0.1 after 500 steps. Output: x_inr.
Stage 2 (slice by slice, zero-shot self-supervised unrolled network, ZS-SSL):
    for every slice, 20 % of the sampled k-space points (outside the 4 x 4 centre) are held out for validation; the rest
    is split 25 times into 60 % data consistency / 40 % loss points (the training set). The network has 10 unrolled
    blocks, each
        x_cnn = CNN([x, reference]) (ResNet, 15 blocks x 64 channels, residual on x)
        x     = argmin ||A_dc x - y||^2 + lambda ||x - x_cnn||^2 (10 CG iterations, lambda learnable, init 0.05),
    started from x = x_inr (unitary FFT). Adam lr 5e-4, batch 1, up to 300 epochs, early stop after 25 epochs without
    validation improvement; the best epoch is used with all sampled points in the data consistency.

Data (MATLAB files; image axes x, y, slice, coils last):
  --kspace (nx, ny, nz, ncoil) complex   --csm coil sensitivities, same size   --tissue-mask (nx, ny, nz)
  --reference registered reference image (nx, ny, nz)   --sampling-mask (nx, ny)   --prior folder with encoder.pt
  --eval-mask optional evaluation mask (nx, ny, n) for the first n slices
Output (--out, .mat): pred_image (MAX, unitary-FFT scaling), x_inr (stage 1, solve scaling), zf_image, SamMask,
  solve_scale, loss_history, stage2_best_epochs, stage2_lambdas, stage2_scale_factors, stage2_val_losses, slices.
  If the k-space is fully sampled, PSNR / SSIM of x_inr and pred_image against the fully sampled root-sum-of-squares image
  are printed and saved (per-slice least-squares intensity matching inside the evaluation mask; slices of --eval-mask).

  python reconstruct_knee.py
  python reconstruct_knee.py --slices 0:18 --out output/recon_Knee_first18.mat

Requires a CUDA GPU, PyTorch, tiny-cuda-nn, numpy, scipy, h5py (scikit-image for SSIM).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from reconstruct import (HASH_GRID, MLP, A_forward, ResBlock2D, build_coords, evaluate, ifft2c, latent_to_voxels,
                         load_mat, tcnn)

HERE = Path(__file__).resolve().parent


@dataclass
class KneeConfig:
    # stage 1
    steps: int = 1000
    lr: float = 1e-3
    latent_lr: float = 1e-4
    n_latent: int = 4
    # stage 2 (per slice)
    epochs: int = 300
    stage2_lr: float = 5e-4
    n_unrolls: int = 10
    n_res_blocks: int = 15
    cg_iter: int = 10
    lambda_init: float = 0.05
    rho_val: float = 0.2
    rho_train: float = 0.4
    num_reps: int = 25
    patience: int = 25
    seed: int = 444
    log_every: int = 100


# --------------------------------------------------------------------------------------
# Stage 1: INR with the prior encoder
# --------------------------------------------------------------------------------------
class KneeINR(nn.Module):
    """Complex image = MLP([prior(coordinate), sum_i w_i(coordinate) * latent_i]) x unit phase(coordinate)."""

    def __init__(self, n_latent: int = 4) -> None:
        super().__init__()
        if tcnn is None:
            raise ImportError("tinycudann is required (https://github.com/NVlabs/tiny-cuda-nn).")
        self.shared_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=HASH_GRID)
        self.latent_encoders = nn.ModuleList([tcnn.Encoding(n_input_dims=2, encoding_config=HASH_GRID) for _ in range(n_latent)])
        self.weight_net = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=n_latent, network_config=MLP,
                                                        encoding_config={"otype": "Frequency", "n_frequencies": 4})
        self.mlp = tcnn.Network(n_input_dims=self.shared_encoder.n_output_dims + self.latent_encoders[0].n_output_dims,
                                n_output_dims=2, network_config=MLP)
        self.phase_net = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=2, network_config=MLP,
                                                       encoding_config={"otype": "Frequency", "n_frequencies": 6})

    def forward(self, coords_01: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        enc = self.shared_encoder(coords_01.float())
        w = torch.softmax(self.weight_net(coords_01.float()), dim=1)
        enc_latent = sum(e(latent[i]) * w[:, i:i + 1] for i, e in enumerate(self.latent_encoders))
        mag = self.mlp(torch.cat([enc, enc_latent], dim=-1)).float()
        out = torch.complex(mag[:, :1], mag[:, 1:])
        ph = self.phase_net(coords_01.float()).float()
        ph = torch.complex(ph[:, :1], ph[:, 1:])
        return out * (ph / torch.sqrt(ph.conj() * ph + 1e-8))

    def load_prior(self, prior_dir: Path) -> None:
        """Only the coordinate encoder comes from the prior (frozen); the latent encoders start from scratch."""
        self.shared_encoder.load_state_dict(torch.load(prior_dir / "encoder.pt", map_location="cpu", weights_only=True))
        for p in self.shared_encoder.parameters():
            p.requires_grad = False


def fit_inr(kspace, sens, mask, tissue, prior_dir: Path, cfg: KneeConfig):
    nx, ny, nz, _ = kspace.shape
    dev = kspace.device
    coords = build_coords(nx, ny, nz, dev)
    latent = nn.Parameter(torch.randn((cfg.n_latent, nx, 1, 1, 2), dtype=torch.float32, device=dev))
    model = KneeINR(cfg.n_latent).to(dev)
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
# Stage 2: per-slice ZS-SSL unrolled network with the reference as CNN input (unitary FFT, numpy data preparation)
# --------------------------------------------------------------------------------------
def ifft_unitary_np(k: np.ndarray) -> np.ndarray:
    n = k.shape[0] * k.shape[1]
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(k, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)) * np.sqrt(n)


def sense_adjoint_np(k: np.ndarray, csm: np.ndarray) -> np.ndarray:
    """(nx, ny, ncoil) k-space -> coil-combined image, unitary FFT."""
    return np.sum(np.conj(csm) * ifft_unitary_np(k), axis=2)


def complex2real(x: np.ndarray) -> np.ndarray:
    return np.stack((x.real, x.imag), axis=-1)


def uniform_selection(kspace: np.ndarray, mask: np.ndarray, rho: float, acs_block: tuple[int, int] = (4, 4)):
    """Draw rho of the sampled points (outside the acs_block centre, uniformly, global numpy RNG) as the loss mask."""
    nrow, ncol = kspace.shape[:2]
    cx = int(np.argsort(np.linalg.norm(np.linalg.norm(kspace, axis=1, keepdims=True), axis=2, keepdims=True).squeeze())[-1])
    cy = int(np.argsort(np.linalg.norm(np.linalg.norm(kspace, axis=0, keepdims=True), axis=2, keepdims=True).squeeze())[-1])
    pr = np.asarray(np.copy(mask), dtype=np.float32)
    pr[cx - acs_block[0] // 2: cx + acs_block[0] // 2, cy - acs_block[1] // 2: cy + acs_block[1] // 2] = 0
    pr = np.maximum(pr.flatten().astype(np.float64), 0)
    ind = np.random.choice(np.arange(nrow * ncol), size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / pr.sum())
    flat = np.zeros(nrow * ncol)
    flat[ind] = 1
    loss_mask = np.zeros_like(mask)
    loss_mask[np.nonzero(flat.reshape(nrow, ncol))] = 1
    return mask - loss_mask, loss_mask


def fft2c_unitary(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"), dim=(-2, -1))


def ifft2c_unitary(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"), dim=(-2, -1))


def conjgrad(rhs_2ch, sens, mask, mu, n_iter: int):
    """Solve (E^H M E + mu) x = rhs for one slice; rhs (2, nx, ny), sens (ncoil, nx, ny), mask (nx, ny)."""
    def inner(a, b):
        return (torch.conj(a) * b).sum().real

    def normal(img):
        k = fft2c_unitary(sens * img) * mask
        return (ifft2c_unitary(k) * torch.conj(sens)).sum(dim=0) + mu * img

    rhs = torch.complex(rhs_2ch[0], rhs_2ch[1])
    mu = mu.type(torch.complex64)
    x = torch.zeros_like(rhs)
    r, p = rhs, rhs
    rsold = inner(r, r)
    for _ in range(n_iter):
        ap = normal(p)
        alpha = rsold / inner(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = inner(r, r)
        p = rsnew / rsold * p + r
        rsold = rsnew
    return torch.stack([x.real, x.imag], dim=0)


def masked_kspace(img_2ch, sens, mask):
    """Image (2, nx, ny) -> k-space at the mask, (ncoil, nx, ny, 2)."""
    k = fft2c_unitary(sens * torch.complex(img_2ch[0], img_2ch[1])) * mask
    return torch.stack([k.real, k.imag], dim=-1)


class GuidedCNN(nn.Module):
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


class GuidedUnrolledSlice(nn.Module):
    def __init__(self, cfg: KneeConfig, guide: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = GuidedCNN(n_blocks=cfg.n_res_blocks)
        self.lam = nn.Parameter(torch.tensor([cfg.lambda_init]))
        self.register_buffer("guide", guide)   # (1, 1, nx, ny)

    def forward(self, atb, x_init, dc_mask, loss_mask, sens):
        """Batch of 1: atb, x_init (1, 2, nx, ny); masks (1, nx, ny); sens (1, ncoil, nx, ny)."""
        x = x_init
        for _ in range(self.cfg.n_unrolls):
            x_cnn = self.cnn(x.float(), self.guide)
            x = conjgrad((atb + self.lam * x_cnn)[0], sens[0], dc_mask[0], self.lam, self.cfg.cg_iter).unsqueeze(0)
        return x, masked_kspace(x[0], sens[0], loss_mask[0]).unsqueeze(0)


def mix_l1_l2(yhat, y):
    return 0.5 * torch.norm(yhat - y) / torch.norm(y) + 0.5 * torch.norm(yhat - y, p=1) / torch.norm(y, p=1)


class SliceSet(Dataset):
    def __init__(self, atb, x_init, dc_masks, loss_masks, sens, ref_k):
        self.items = (atb, x_init, dc_masks, loss_masks, sens, ref_k)

    def __len__(self):
        return len(self.items[0])

    def __getitem__(self, i):
        atb, x0, dcm, lm, s, rk = (a[i] for a in self.items)
        return (torch.tensor(atb, dtype=torch.float64), torch.tensor(x0, dtype=torch.float64),
                torch.tensor(dcm, dtype=torch.complex64), torch.tensor(lm, dtype=torch.complex64),
                torch.tensor(s, dtype=torch.complex64), torch.tensor(rk, dtype=torch.float64))


def refine_slice(k_slice, csm_slice, mask2d, x_inr_slice, guide_slice, cfg: KneeConfig, dev):
    """Train the guided unrolled network on one slice (ZS-SSL) and return the reconstruction (unitary scaling)."""
    scale = np.float32(np.max(np.abs(k_slice))) if np.max(np.abs(k_slice)) > 0 else np.float32(1.0)
    k = k_slice / scale
    x0 = x_inr_slice / scale
    nx, ny, nc = k.shape

    def coil_mask(m):
        return np.tile(m[:, :, None], (1, 1, nc))

    remainder, val_mask = uniform_selection(k, mask2d, cfg.rho_val)
    remainder, val_mask = remainder.astype(np.float32), val_mask.astype(np.float32)
    dc_masks = np.empty((cfg.num_reps, nx, ny), dtype=np.float32)
    loss_masks = np.empty_like(dc_masks)
    atb_trn = np.empty((cfg.num_reps, nx, ny), dtype=np.complex64)
    ref_trn = np.empty((cfg.num_reps, nx, ny, nc), dtype=np.complex64)
    for j in range(cfg.num_reps):
        dc_masks[j], loss_masks[j] = uniform_selection(k, remainder, cfg.rho_train)
        atb_trn[j] = sense_adjoint_np(k * coil_mask(dc_masks[j]), csm_slice)
        ref_trn[j] = k * coil_mask(loss_masks[j])
    sens_t = np.transpose(np.tile(csm_slice[None], (cfg.num_reps, 1, 1, 1)), (0, 3, 1, 2))
    train = DataLoader(SliceSet(complex2real(atb_trn), complex2real(np.tile(x0[None], (cfg.num_reps, 1, 1)).astype(np.complex64)),
                                dc_masks, loss_masks, sens_t, complex2real(np.transpose(ref_trn, (0, 3, 1, 2)))),
                       batch_size=1, shuffle=True, num_workers=0)
    val = DataLoader(SliceSet(complex2real(sense_adjoint_np(k * coil_mask(remainder), csm_slice)[None]), complex2real(x0[None].copy()),
                              remainder[None], val_mask[None], sens_t[:1],
                              complex2real(np.transpose((k * coil_mask(val_mask))[None], (0, 3, 1, 2)))),
                     batch_size=1, shuffle=False, num_workers=0)

    net = GuidedUnrolledSlice(cfg, torch.from_numpy(guide_slice[None, None]).float().to(dev)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.stage2_lr)

    def step(batch, train_mode):
        atb, xi, dcm, lm, s, rk = (b.to(dev) for b in batch)
        _, kout = net(atb.permute(0, 3, 1, 2), xi.permute(0, 3, 1, 2), dcm, lm, s)
        return mix_l1_l2(kout, rk)

    best, best_ep, bad, best_state, val_hist, ep = np.inf, -1, 0, None, [], 0
    while ep < cfg.epochs and bad < cfg.patience:
        net.train()
        for batch in train:
            loss = step(batch, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean([step(b, False).item() for b in val]))
        val_hist.append(v)
        if v < best:
            best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
        ep += 1
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():   # all sampled points in the data consistency
        atb = torch.from_numpy(complex2real(sense_adjoint_np(k * coil_mask(mask2d), csm_slice))[None]).permute(0, 3, 1, 2).to(dev)
        xi = torch.from_numpy(complex2real(x0)[None]).permute(0, 3, 1, 2).to(dev)
        m = torch.from_numpy(mask2d[None]).to(dev)
        s = torch.from_numpy(np.transpose(csm_slice[None], (0, 3, 1, 2))).to(dev)
        x, _ = net(atb, xi, m, m, s)
    x = x[0].cpu().numpy()
    return (x[0] + 1j * x[1]) * scale, best_ep, float(net.lam.item()), float(scale), np.asarray(val_hist, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = HERE / "data"
    p.add_argument("--kspace", type=Path, default=d / "Knee_T2w_kspace.mat")
    p.add_argument("--kspace-key", default="T2w_kspace")
    p.add_argument("--csm", type=Path, default=d / "Knee_T1w2T2w_tstCsm.mat")
    p.add_argument("--csm-key", default="T1w2T2w_tstCsm")
    p.add_argument("--tissue-mask", type=Path, default=d / "Knee_T2w_tissue_mask.mat")
    p.add_argument("--tissue-mask-key", default="tissue_mask")
    p.add_argument("--reference", type=Path, default=d / "Knee_T1w2T2w.mat")
    p.add_argument("--reference-key", default="T1w2T2w")
    p.add_argument("--sampling-mask", type=Path, default=d / "Knee_SamMask.mat")
    p.add_argument("--sampling-mask-key", default="SamMask")
    p.add_argument("--eval-mask", type=Path, default=d / "Knee_eval_mask.mat", help="evaluation mask for the first n slices")
    p.add_argument("--eval-mask-key", default="eval_mask")
    p.add_argument("--prior", type=Path, default=HERE.parent / "prior_embedding" / "output_Knee")
    p.add_argument("--slices", default=None, help="stage-2 slices as start:stop (default all); others stay 0")
    p.add_argument("--out", type=Path, default=HERE / "output" / "recon_Knee.mat")
    p.add_argument("--gpu-id", type=int, default=0)
    a = p.parse_args()

    cfg = KneeConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (tiny-cuda-nn).")
    torch.cuda.set_device(a.gpu_id)
    dev = torch.device(f"cuda:{a.gpu_id}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    kspace_np = load_mat(a.kspace, a.kspace_key).astype(np.complex64)
    csm_np = load_mat(a.csm, a.csm_key).astype(np.complex64)
    tissue_np = (load_mat(a.tissue_mask, a.tissue_mask_key) > 0).astype(np.float32)
    ref_np = np.abs(load_mat(a.reference, a.reference_key)).astype(np.float32)
    nx, ny, nz, nc = kspace_np.shape
    if csm_np.shape != kspace_np.shape or tissue_np.shape != (nx, ny, nz) or ref_np.shape != (nx, ny, nz):
        raise ValueError(f"shapes: kspace {kspace_np.shape}, csm {csm_np.shape}, tissue {tissue_np.shape}, reference {ref_np.shape}")
    sam = (np.real(load_mat(a.sampling_mask, a.sampling_mask_key)) > 0).astype(np.float32)
    s0, s1 = (int(v) for v in a.slices.split(":")) if a.slices else (0, nz)
    print(f"k-space {kspace_np.shape}, R = {sam.size / sam.sum():.2f}, prior {a.prior}, stage-2 slices {s0}:{s1}", flush=True)

    kspace = torch.from_numpy(kspace_np).to(dev)
    sens = torch.from_numpy(csm_np).to(dev)
    tissue = torch.from_numpy(tissue_np).to(dev)
    mask = torch.from_numpy(sam > 0).to(dev)[:, :, None, None].expand(nx, ny, nz, nc)
    k_meas = kspace * mask
    zf = torch.sqrt((ifft2c(k_meas).abs() ** 2).sum(dim=3))
    scale = zf.max().clamp_min(1e-8)

    t0 = time.time()
    x_inr, loss_hist = fit_inr(k_meas / scale, sens, mask, tissue, a.prior, cfg)
    x_inr_np = (x_inr * scale).cpu().numpy().astype(np.complex64)
    del kspace, sens, mask, k_meas
    torch.cuda.empty_cache()
    t1 = time.time()
    print(f"stage 1 {(t1 - t0) / 60:.1f} min", flush=True)

    k_meas_np = kspace_np * sam[:, :, None, None]
    guide_np = ref_np * tissue_np
    pred = np.zeros((nx, ny, nz), dtype=np.complex64)
    best_eps, lams, scales = (np.full(nz, -1, np.int32), np.zeros(nz, np.float32), np.zeros(nz, np.float32))
    val_hists = []
    for z in range(s0, s1):
        ts = time.time()
        x0 = np.sqrt(nx * ny) * x_inr_np[:, :, z]   # solve scaling (non-unitary FFT) -> unitary FFT scaling
        pred[:, :, z], best_eps[z], lams[z], scales[z], vh = refine_slice(k_meas_np[:, :, z], csm_np[:, :, z], sam, x0,
                                                                           guide_np[:, :, z], cfg, dev)
        val_hists.append(vh)
        print(f"[stage 2] slice {z + 1}/{nz}  best epoch {best_eps[z] + 1}  lambda {lams[z]:.4f}  {(time.time() - ts) / 60:.1f} min", flush=True)
    t2 = time.time()
    print(f"stage 2 {(t2 - t1) / 60:.1f} min", flush=True)

    out = {"pred_image": pred, "x_inr": x_inr_np, "zf_image": zf.cpu().numpy().astype(np.float32), "SamMask": sam,
           "solve_scale": float(scale), "loss_history": loss_hist, "stage2_best_epochs": best_eps, "stage2_lambdas": lams,
           "stage2_scale_factors": scales, "stage2_val_losses": np.array(val_hists, dtype=object),
           "slices": np.arange(s0, s1), "minutes": (t2 - t0) / 60, "config": json.dumps(asdict(cfg))}
    if np.count_nonzero(np.abs(kspace_np[:, :, nz // 2, 0]).sum(0)) == ny:   # fully sampled: evaluate
        rss = np.sqrt((np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_np, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))) ** 2).sum(-1))
        if a.eval_mask and a.eval_mask.exists():
            em = (load_mat(a.eval_mask, a.eval_mask_key) > 0).astype(np.float32)
        else:
            em = tissue_np
        n = em.shape[2]
        if s0 > 0 or s1 < n:
            print(f"note: evaluation uses slices 0:{n}, stage 2 ran on {s0}:{s1}", flush=True)
        metrics = {k: evaluate(v[:, :, :n], rss[:, :, :n].astype(np.float32), em) for k, v in (("x_inr", x_inr_np), ("pred_image", pred))}
        print("metrics vs fully sampled:", json.dumps(metrics), flush=True)
        out["metrics"] = json.dumps(metrics)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(a.out), out, do_compression=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
