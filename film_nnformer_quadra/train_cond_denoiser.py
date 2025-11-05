#!/usr/bin/env python3
"""
Training script for a FiLM-conditioned nnFormer denoiser.
- Reads data from a Zarr store with structure:
    /RAID/mpet/PET_LOWDOSE/TRAINING_DATA/quadra_pet.zarr
        <case_id>/
            clean         (D,H,W)
            noisy_<level> (D,H,W)  # e.g., noisy_0.050, noisy_0.100, ...
- Samples random 3D patches and trains to regress clean from noisy, conditioned on the noise level.
- Requires: your nnFormer repo patched to accept `forward(x, cond)` where cond is [B, d_cond].
"""
import os
import math
import time
import argparse
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import zarr

try:
    from monai.losses import SSIMLoss
except Exception:
    SSIMLoss = None  # will gate usage below

# --- Utilities ----------------------------------------------------------------

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sigma_from_key(key: str) -> float:
    """
    Accepts dataset keys like 'noisy_0.200' or 'noisy_1-10' (interpreted as 1/10).
    Returns the numeric noise level as float.
    """
    suffix = str(key).split("_", 1)[1]
    suffix = suffix.replace(",", ".")
    if "-" in suffix:
        num, denom = suffix.split("-", 1)
        try:
            return float(num) / float(denom)
        except Exception as exc:
            raise ValueError(f"Cannot parse sigma from key '{key}' as fraction") from exc
    try:
        return float(suffix)
    except Exception as exc:
        raise ValueError(f"Cannot parse sigma from key '{key}'") from exc

def discover_cases(zarr_path: str) -> List[str]:
    root = zarr.open(zarr_path, mode="r")
    # Try to use group_keys (zarr>=2) else fallback
    try:
        keys = list(root.group_keys())
    except Exception:
        keys = list(root.keys())
    # Filter groups that contain a "clean" dataset
    cases = []
    for k in keys:
        try:
            g = root[k]
            if isinstance(g, zarr.hierarchy.Group) and "clean" in g.keys():
                cases.append(k)
        except Exception:
            pass
    if not cases:
        # As a fallback, treat every top-level key as a case if it has "clean"
        for k in keys:
            try:
                if "clean" in root[k].keys():
                    cases.append(k)
            except Exception:
                continue
    if not cases:
        raise RuntimeError("No cases with a 'clean' dataset were found in the Zarr store.")
    return sorted(cases)

def discover_sigma_levels(zarr_path: str, cases: List[str], include_clean: bool = True) -> Tuple[float, float]:
    root = zarr.open(zarr_path, mode="r")
    sigmas = []
    for cid in cases:
        try:
            g = root[cid]
            for k in g.keys():
                if str(k).startswith("noisy_"):
                    try:
                        sigmas.append(parse_sigma_from_key(k))
                    except Exception:
                        pass
        except Exception:
            pass
    if include_clean:
        sigmas.append(1.0)
    if not sigmas:
        raise RuntimeError("No noisy_* datasets found to infer sigma range.")
    return float(np.min(sigmas)), float(np.max(sigmas))

def build_items_and_shapes(zarr_path: str, cases: List[str], include_clean: bool = True) -> Tuple[List[Tuple[str, str, float]], Dict[str, Tuple[int,int,int]]]:
    """
    Returns:
      items: list of (case_id, noisy_key, sigma_float)
      shapes: mapping case_id -> clean shape (D,H,W)
    """
    root = zarr.open(zarr_path, mode="r")
    items = []
    shapes = {}
    for cid in cases:
        g = root[cid]
        if "clean" not in g:
            continue
        shapes[cid] = tuple(int(x) for x in g["clean"].shape)
        for k in g.keys():
            if str(k).startswith("noisy_"):
                try:
                    s = parse_sigma_from_key(k)
                    items.append((cid, str(k), s))
                except Exception:
                    continue
        if include_clean:
            items.append((cid, "clean", 1.0))
    if not items:
        raise RuntimeError("No (case, noisy_*) pairs found.")
    return items, shapes

# --- Dataset ------------------------------------------------------------------

class ZarrDenoisePatches(Dataset):
    def __init__(
        self,
        zarr_path: str,
        cases: List[str],
        items: List[Tuple[str, str, float]],
        shapes: Dict[str, Tuple[int,int,int]],
        sigma_min: float,
        sigma_max: float,
        patch: Tuple[int,int,int] = (96,96,96),
        samples_per_vol: int = 64,
        normalize_intensity: bool = True,
        include_clean: bool = True,
    ):
        self.zarr_path = zarr_path
        self.cases = cases
        self.items = items
        self.shapes = shapes
        self.patch = patch
        self.samples_per_vol = samples_per_vol
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.normalize_intensity = normalize_intensity
        self.include_clean = include_clean

    def __len__(self) -> int:
        return len(self.items) * self.samples_per_vol

    def _rand_crop(self, shp: Tuple[int,int,int]) -> Tuple[slice, slice, slice]:
        D,H,W = shp
        pd,ph,pw = self.patch
        z = random.randint(0, max(0, D-pd)) if D >= pd else 0
        y = random.randint(0, max(0, H-ph)) if H >= ph else 0
        x = random.randint(0, max(0, W-pw)) if W >= pw else 0
        return slice(z, z+pd), slice(y, y+ph), slice(x, x+pw)

    def __getitem__(self, index: int):
        cid, noisy_key, sigma = self.items[index % len(self.items)]
        # Open store inside the worker for process-safety
        root = zarr.open(self.zarr_path, mode="r")
        g = root[cid]
        clean = g["clean"]
        noisy = clean if noisy_key == "clean" else g[noisy_key]

        slz, sly, slx = self._rand_crop(self.shapes[cid])
        n = np.asarray(noisy[slz, sly, slx], dtype=np.float32)
        c = np.asarray(clean[slz, sly, slx], dtype=np.float32)

        # Replace NaN/Inf with finite numbers
        n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_intensity:
            stacked = np.concatenate([n[None], c[None]], axis=0)
            vmin = float(stacked.min())
            vmax = float(stacked.max())
            denom = vmax - vmin
            if denom > 0:
                n = (n - vmin) / denom
                c = (c - vmin) / denom
            else:
                n = np.zeros_like(n, dtype=np.float32)
                c = np.zeros_like(c, dtype=np.float32)

        n = n.astype(np.float32)[None]  # [1,D,H,W]
        c = c.astype(np.float32)[None]
        sigma_val = 1.0 if noisy_key == "clean" else float(sigma)
        sigma_vec = np.array([sigma_val], dtype=np.float32)

        # Optional: intensity normalization can be added here

        return torch.from_numpy(n), torch.from_numpy(c), torch.from_numpy(sigma_vec)

# --- Model wrapper ------------------------------------------------------------

def import_nnformer():
    """
    Try a couple of common import paths for nnFormer.
    You must have patched the model so that forward(x, cond) is supported.
    """
    tried = []
    try:
        from nnformer.network_architecture.nnFormer_synapse import nnFormer as NNFormer
        return NNFormer
    except Exception as e:
        tried.append(("nnformer.network_architecture.nnFormer_synapse", str(e)))
    try:
        from nnformer.network_architecture.nnFormer import nnFormer as NNFormer
        return NNFormer
    except Exception as e:
        tried.append(("nnformer.network_architecture.nnFormer", str(e)))
    msg = "Could not import nnFormer. Tried:\n" + "\n".join([f"  {m}: {err}" for m,err in tried])
    raise ImportError(msg)

class CondNNFormerDenoiser(nn.Module):
    def __init__(self, crop_size: Tuple[int, int, int], d_cond: int = 1):
        super().__init__()
        NNFormer = import_nnformer()
        # Ensure your patched NNFormer accepts in_channels=1, num_classes=1 and forward(x, cond)
        self.net = NNFormer(
            crop_size=list(crop_size),
            input_channels=1,
            num_classes=1,
            deep_supervision=False,
            d_cond=d_cond,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.net(x, cond)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out

# --- Training loop ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="/RAID/mpet/PET_LOWDOSE/TRAINING_DATA/quadra_pet.zarr", help="Path to Zarr dataset")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patch", type=str, default="96,96,96", help="Crop size D,H,W used for training")
    ap.add_argument("--samples_per_vol", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--save", default="cond_nnformer_denoiser_best.pt")
    ap.add_argument("--no_ssim", action="store_true", help="Disable SSIM component of the loss")
    ap.add_argument("--no_intensity_norm", action="store_true", help="Disable per-patch min-max intensity normalization")
    ap.add_argument("--disable_amp", action="store_true", help="Disable autocast/GradScaler mixed precision even on CUDA")
    ap.add_argument("--no_clean_sigma", action="store_true", help="Do not duplicate the clean reference as a sigma=1.0 training sample")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    patch = tuple(int(x) for x in args.patch.split(","))
    cases = discover_cases(args.zarr)
    include_clean = not args.no_clean_sigma
    sigma_min, sigma_max = discover_sigma_levels(args.zarr, cases, include_clean=include_clean)
    items, shapes = build_items_and_shapes(args.zarr, cases, include_clean=include_clean)

    # simple 90/10 split by case
    split = int(0.9 * len(cases)) if len(cases) > 1 else 1
    train_cases = cases[:split]
    val_cases   = cases[split:] if split < len(cases) else cases[:1]

    # Filter items by split
    train_items = [it for it in items if it[0] in train_cases]
    val_items   = [it for it in items if it[0] in val_cases]

    train_ds = ZarrDenoisePatches(args.zarr, train_cases, train_items, shapes, sigma_min, sigma_max,
                                  patch=patch, samples_per_vol=args.samples_per_vol,
                                  normalize_intensity=not args.no_intensity_norm,
                                  include_clean=include_clean)
    val_ds   = ZarrDenoisePatches(args.zarr, val_cases,   val_items,   shapes, sigma_min, sigma_max,
                                  patch=patch, samples_per_vol=max(8, args.samples_per_vol//4),
                                  normalize_intensity=not args.no_intensity_norm,
                                  include_clean=include_clean)

    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                          pin_memory=True, shuffle=True, persistent_workers=(args.workers>0))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, num_workers=max(2, args.workers//2),
                          pin_memory=True, shuffle=False, persistent_workers=(args.workers>0))

    print(f"Prepared datasets → train_cases={len(train_cases)} ({len(train_ds)} samples) | "
          f"val_cases={len(val_cases)} ({len(val_ds)} samples)", flush=True)
    print(f"Train loader batches/epoch: {len(train_dl)} | Val batches: {len(val_dl)}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CondNNFormerDenoiser(crop_size=patch, d_cond=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    l1 = nn.SmoothL1Loss(beta=0.01)
    use_ssim = (SSIMLoss is not None) and (not args.no_ssim)
    if use_ssim:
        ssim = SSIMLoss(spatial_dims=3)
    else:
        ssim = None

    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best = math.inf

    log_every = max(10, min(500, max(1, len(train_dl) // 100)))
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        for step, (noisy, clean, sigma) in enumerate(train_dl, 1):
            noisy, clean, sigma = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True), sigma.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(noisy, sigma)  # sigma is [B,1] normalized to [0,1]
                loss = 0.85 * l1(pred, clean)
                if ssim is not None:
                    loss = loss + 0.15 * ssim(pred, clean)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                opt.step()

            if step % log_every == 0 or step == 1:
                print(f"Epoch {epoch:03d} | step {step:04d}/{len(train_dl)} | loss {float(loss.detach()):.5f}", flush=True)

        # Validation (L1 only for speed)
        model.eval()
        vl = 0.0; vc = 0
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            for noisy, clean, sigma in val_dl:
                pred = model(noisy.to(device, non_blocking=True), sigma.to(device, non_blocking=True))
                vl += l1(pred, clean.to(device, non_blocking=True)).item()
                vc += 1
        vl = vl / max(vc, 1)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), args.save)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d} complete in {epoch_time:.1f}s | val_L1={vl:.6f} | best={best:.6f} | saved={vl<=best}", flush=True)

if __name__ == "__main__":
    main()
