#!/usr/bin/env python3
"""
Minimal MONAI DynUNet training + inference script for low-dose PET.

Assumptions:
- Input data is located under:
  /root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/all_subjects
  Each patient folder contains two subfolders:
    - "1-10 dose"  or other DRF (input)
    - "Full_dose"  (target)
- Each subfolder contains a DICOM series (Siemens .IMA files). We read the first series found.
- Outputs (.nii.gz) are written to:
  /root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output

This is a minimal prototype script with light error handling.
"""

import os
import sys
import math
import random
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import SimpleITK as sitk
except Exception as e:
    print("SimpleITK is required but not installed.", file=sys.stderr)
    raise

try:
    from monai.networks.nets import DynUNet
    from monai.losses import SSIMLoss
    from monai.inferers import SlidingWindowInferer
except Exception as e:
    print("MONAI is required but not installed.", file=sys.stderr)
    raise


# ----------------------------- Config ---------------------------------

DATA_ROOT = "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/all_subjects"
OUTPUT_ROOT = "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output"

# Local artifact directories
BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "trained-models")
RUN_ID = time.strftime("%Y%m%d_%H%M")  # updated later to include DRF
RUN_ID_BASE = RUN_ID

PATCH_SIZE = (96, 96, 96)  ###(80, 80, 80)
# Slight overlap: stride < patch size; 80 gives 16 voxels overlap
PATCH_STRIDE = (80, 80, 80)  ###(64, 64, 64)

MAX_PATIENTS = 3  ###None  # set to an int to limit for quick tests
VAL_FRACTION = 0.2
RANDOM_SEED = 42

BATCH_SIZE = 8
EPOCHS = 2  ###10
LR = 1e-4
WEIGHT_DECAY = 1e-5

LOSS_MSE_W = 0.9
LOSS_SSIM_W = 0.1

CURVE_PNG = os.path.join(FIG_DIR, f"training_curves_{RUN_ID}.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ----------------------------- I/O Utils -------------------------------

def read_dicom_series(folder: str) -> Tuple[np.ndarray, Dict]:
    """Read first DICOM series in a folder using SimpleITK.

    Returns:
        vol: numpy array shaped [D, H, W]
        meta: dict with keys spacing, origin, direction
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {folder}")
    series_files = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
    reader.SetFileNames(series_files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)  # [D, H, W]
    meta = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
    }
    return arr, meta


def write_nifti(volume: np.ndarray, meta: Dict, out_path: str):
    """Write [D, H, W] volume to NIfTI with given metadata."""
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    # Assign spacing/origin/direction if present
    if meta and "spacing" in meta:
        try:
            img.SetSpacing(meta["spacing"])
        except Exception:
            pass
    if meta and "origin" in meta:
        try:
            img.SetOrigin(meta["origin"])
        except Exception:
            pass
    if meta and "direction" in meta:
        try:
            img.SetDirection(meta["direction"])
        except Exception:
            pass
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path)


# --------------------------- Data Handling -----------------------------

@dataclass
class PatientItem:
    pid: str
    input_arr: np.ndarray  # [D,H,W]
    target_arr: np.ndarray  # [D,H,W]
    input_meta: Dict
    target_meta: Dict


def zscore(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-6:
        return x * 0.0
    return (x - m) / s


def find_patients(data_root: str) -> List[str]:
    pids = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if os.path.isdir(p):
            pids.append(name)
    return pids


def load_patient(data_root: str, pid: str, drf: int) -> PatientItem:
    in_dir = os.path.join(data_root, pid, f"1-{drf} dose")
    tgt_dir = os.path.join(data_root, pid, "Full_dose")
    in_vol, in_meta = read_dicom_series(in_dir)
    tgt_vol, tgt_meta = read_dicom_series(tgt_dir)

    # Simple size alignment if small mismatches; crop to min dims
    dz = min(in_vol.shape[0], tgt_vol.shape[0])
    dy = min(in_vol.shape[1], tgt_vol.shape[1])
    dx = min(in_vol.shape[2], tgt_vol.shape[2])
    in_vol = in_vol[:dz, :dy, :dx]
    tgt_vol = tgt_vol[:dz, :dy, :dx]

    # Normalize (z-score per-volume)
    in_vol = zscore(in_vol).astype(np.float32, copy=False)
    tgt_vol = zscore(tgt_vol).astype(np.float32, copy=False)

    return PatientItem(pid=pid, input_arr=in_vol, target_arr=tgt_vol, input_meta=in_meta, target_meta=tgt_meta)


def compute_grid(starts: int, size: int, step: int) -> List[int]:
    pts = list(range(0, max(1, starts - size + 1), step))
    if not pts or pts[-1] != max(0, starts - size):
        pts.append(max(0, starts - size))
    return pts


def enumerate_patches(shape: Tuple[int, int, int], patch_size: Tuple[int, int, int], stride: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    d, h, w = shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    z_list = compute_grid(d, pd, sd)
    y_list = compute_grid(h, ph, sh)
    x_list = compute_grid(w, pw, sw)
    coords = []
    for z in z_list:
        for y in y_list:
            for x in x_list:
                coords.append((z, y, x))
    return coords


def random_augment(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x,y: [C,D,H,W]
    # Random flips along D/H/W
    for dim in [1, 2, 3]:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[dim])
            y = torch.flip(y, dims=[dim])
    # Random 90-degree rotations in H/W plane
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
        y = torch.rot90(y, k, dims=(2, 3))
    # Random 90-degree rotations in D/H plane
    if random.random() < 0.5:
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(1, 2))
    return x, y


class PatchDataset(Dataset):
    def __init__(self, patients: List[PatientItem], patch_size: Tuple[int, int, int], stride: Tuple[int, int, int], augment: bool = True):
        self.patients = patients
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment

        # Precompute patch coordinates for all patients
        self.index: List[Tuple[int, Tuple[int, int, int]]] = []  # (patient_idx, (z,y,x))
        for pi, p in enumerate(self.patients):
            coords = enumerate_patches(p.input_arr.shape, patch_size, stride)
            for c in coords:
                self.index.append((pi, c))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pi, (z, y, x) = self.index[i]
        p = self.patients[pi]
        pd, ph, pw = self.patch_size

        xin = p.input_arr[z:z+pd, y:y+ph, x:x+pw]
        ygt = p.target_arr[z:z+pd, y:y+ph, x:x+pw]

        # to tensors, add channel dim: [C,D,H,W]
        xt = torch.from_numpy(xin).float().unsqueeze(0)
        yt = torch.from_numpy(ygt).float().unsqueeze(0)

        if self.augment:
            xt, yt = random_augment(xt, yt)

        return xt, yt


# ----------------------------- Model -----------------------------------

def make_model() -> nn.Module:
    kernel_size = [[3, 3, 3]] * 5
    strides = [1, 2, 2, 2, 2]
    up_kernels = [2, 2, 2, 2]
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=up_kernels,
        norm_name=("instance", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=False,
    )
    return model


# ---------------------------- Training ---------------------------------

def plot_curves(train_hist: List[float], val_hist: List[float], out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def train_and_validate(train_ds: PatchDataset, val_patients: List[PatientItem]):
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = make_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    mse = nn.MSELoss()
    ssim = SSIMLoss(spatial_dims=3, data_range=2.0)  # z-scored; approximate range

    # MONAI sliding window inferer for validation/inference
    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.2, mode="gaussian")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, 
                        pin_memory=(DEVICE.type == "cuda"), persistent_workers=True)

    train_hist: List[float] = []
    val_hist: List[float] = []

    # Track and optionally print GPU memory usage once during training
    printed_gpu_mem = False
    if DEVICE.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        skipped_empty = 0
        total_seen = 0
        t0 = time.time()
        for xb, yb in tqdm(loader, desc="Training", unit="batch"):
            xb = xb.to(DEVICE, non_blocking=True)  # [B,C,D,H,W]
            yb = yb.to(DEVICE)

            # # Skip patches that are practically empty (per-sample within batch)
            # with torch.no_grad():
            #     flat = xb.abs().view(xb.shape[0], -1)
            #     p95 = torch.quantile(flat, 0.95, dim=1)
            #     thresh = 0.02 * p95
            #     meanv = flat.mean(dim=1)
            #     stdv = flat.std(dim=1, unbiased=False)
            #     keep = ~((meanv <= thresh) & (stdv <= thresh))
            # total_seen += xb.shape[0]
            # if keep.sum().item() == 0:
            #     skipped_empty += xb.shape[0]
            #     continue
            # if keep.sum().item() < xb.shape[0]:
            #     skipped_empty += int((~keep).sum().item())
            #     xb = xb[keep]
            #     yb = yb[keep]
                
            opt.zero_grad()
            pred = model(xb)
            l = LOSS_MSE_W * mse(pred, yb) ###+ LOSS_SSIM_W * ssim(pred, yb)
            l.backward()
            opt.step()
            ep_loss += float(l.item())
            n_batches += 1

            # Print GPU memory usage once (after first optimization step)
            if (not printed_gpu_mem) and DEVICE.type == "cuda":
                try:
                    torch.cuda.synchronize()
                    dev_idx = torch.cuda.current_device()
                    props = torch.cuda.get_device_properties(dev_idx)
                    alloc = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
                    max_alloc = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
                    total = getattr(props, 'total_memory', 0) / (1024 ** 3)
                    print(f"GPU memory: alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={max_alloc:.2f}GB total={total:.2f}GB")
                except Exception:
                    pass
                printed_gpu_mem = True
        train_loss = ep_loss / max(1, n_batches)
        train_hist.append(train_loss)

        # Validation: full-volume sliding window on val patients
        model.eval()
        with torch.no_grad():
            vloss_acc = 0.0
            vcount = 0
            for vp in val_patients:
                xt = torch.from_numpy(vp.input_arr).float()[None, None]  # [1,1,D,H,W]
                yt = torch.from_numpy(vp.target_arr).float()[None, None]
                xt = xt.to(DEVICE)
                yt = yt.to(DEVICE)
                pred_full = inferer(xt, model)
                lv = LOSS_MSE_W * mse(pred_full, yt) + LOSS_SSIM_W * ssim(pred_full, yt)
                vloss_acc += float(lv.item())
                vcount += 1
            val_loss = vloss_acc / max(1, vcount)
            val_hist.append(val_loss)

        dt = time.time() - t0
        skip_msg = f" | skipped_empty={skipped_empty}/{total_seen}" if total_seen > 0 else ""
        print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.4f} val={val_loss:.4f} ({dt:.1f}s){skip_msg}")
        try:
            plot_curves(train_hist, val_hist, CURVE_PNG)
        except Exception:
            pass

    return model, inferer, train_hist, val_hist


# --------------------------- Evaluation ---------------------------------

def mse_ssim_metrics(pred: torch.Tensor, target: torch.Tensor, ssim_loss: SSIMLoss) -> Tuple[float, float]:
    # tensors are [1,1,D,H,W]
    with torch.no_grad():
        m = float(F.mse_loss(pred, target).item())
        # SSIMLoss returns (1 - ssim), so true SSIM = 1 - loss
        sl = float(ssim_loss(pred, target).item())
        ssim_val = 1.0 - sl
        return m, ssim_val


def save_all_outputs(model: nn.Module, inferer: SlidingWindowInferer, patients: List[PatientItem]):
    model.eval()
    ssim_eval = SSIMLoss(spatial_dims=3, data_range=2.0)
    rows = []
    with torch.no_grad():
        for p in patients:
            xt = torch.from_numpy(p.input_arr).float()[None, None].to(DEVICE)
            yt = torch.from_numpy(p.target_arr).float()[None, None].to(DEVICE)
            pred = inferer(xt, model)

            # Metrics vs target (model) and baseline (input)
            m_mse, m_ssim = mse_ssim_metrics(pred, yt, ssim_eval)
            b_mse, b_ssim = mse_ssim_metrics(xt, yt, ssim_eval)

            # Save NIfTI
            pred_np = pred[0, 0].cpu().numpy()
            out_path = os.path.join(OUTPUT_ROOT, f"{p.pid}.nii.gz")
            try:
                write_nifti(pred_np, p.input_meta, out_path)
            except Exception:
                # fallback without metadata
                write_nifti(pred_np, {}, out_path)

            print(f"Saved: {out_path} | model MSE={m_mse:.4f} SSIM={m_ssim:.4f} | baseline MSE={b_mse:.4f} SSIM={b_ssim:.4f}")
            rows.append((p.pid, m_mse, m_ssim, b_mse, b_ssim))

    # Write a small summary CSV alongside outputs
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_path = os.path.join(OUTPUT_ROOT, f"metrics_summary_{RUN_ID}.csv")
    try:
        with open(summary_path, "w") as f:
            f.write("patient,model_mse,model_ssim,baseline_mse,baseline_ssim\n")
            for r in rows:
                f.write(",".join([r[0]] + [f"{x:.6f}" for x in r[1:]]) + "\n")
        print(f"Wrote metrics summary: {summary_path}")
    except Exception:
        pass


# ------------------------------ Main -----------------------------------

def plot_val_examples(model: nn.Module, inferer: SlidingWindowInferer, val_patients: List[PatientItem], n: int = 3):
    """Plot up to n coronal-slice examples from validation set.
    Layout: top row = input/target/output (inferno); bottom row = error maps
    for target-input and target-output (bwr). Saves PNGs in FIG_DIR.
    """
    if not val_patients:
        print("No validation patients available for plotting.")
        return
    model.eval()
    os.makedirs(FIG_DIR, exist_ok=True)
    count = min(n, len(val_patients))
    with torch.no_grad():
        for p in val_patients[:count]:
            xt = torch.from_numpy(p.input_arr).float()[None, None].to(DEVICE)
            yt = torch.from_numpy(p.target_arr).float()[None, None].to(DEVICE)
            pred = inferer(xt, model)
            xnp = xt[0, 0].cpu().numpy()
            ynp = yt[0, 0].cpu().numpy()
            pnp = pred[0, 0].cpu().numpy()

            # Coronal slice: fix X (W axis) at middle
            wmid = xnp.shape[2] // 2
            x_cor = xnp[:, :, wmid]
            y_cor = ynp[:, :, wmid]
            p_cor = pnp[:, :, wmid]

            # Common intensity range for images
            vmin = float(min(x_cor.min(), y_cor.min(), p_cor.min()))
            vmax = float(max(x_cor.max(), y_cor.max(), p_cor.max()))

            # Errors
            e_in = y_cor - x_cor
            e_out = y_cor - p_cor
            eabs = float(max(abs(e_in).max(), abs(e_out).max(), 1e-6))

            # Build a compact 2x3 grid (bottom-right left blank)
            fig = plt.figure(figsize=(10, 6), dpi=120)
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.02, hspace=0.08)
            ax_in = fig.add_subplot(gs[0, 0])
            ax_tg = fig.add_subplot(gs[0, 1])
            ax_out = fig.add_subplot(gs[0, 2])
            ax_ein = fig.add_subplot(gs[1, 0])
            ax_eout = fig.add_subplot(gs[1, 1])
            ax_empty = fig.add_subplot(gs[1, 2])

            ax_in.imshow(x_cor.T, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
            ax_in.set_title('Input')
            ax_tg.imshow(y_cor.T, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
            ax_tg.set_title('Target')
            ax_out.imshow(p_cor.T, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
            ax_out.set_title('Output')
            ax_ein.imshow(e_in.T, cmap='bwr', vmin=-eabs, vmax=eabs, origin='lower')
            ax_ein.set_title('Err T-Input')
            ax_eout.imshow(e_out.T, cmap='bwr', vmin=-eabs, vmax=eabs, origin='lower')
            ax_eout.set_title('Err T-Output')
            for ax in (ax_in, ax_tg, ax_out, ax_ein, ax_eout, ax_empty):
                ax.axis('off')
            ax_empty.set_visible(False)
            fig.suptitle(p.pid)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_png = os.path.join(FIG_DIR, f"{RUN_ID}_{p.pid}_coronal.png")
            plt.savefig(out_png)
            plt.close(fig)
            print(f"Saved validation example plot: {out_png}")


def infer_full_volume(x: torch.Tensor, model: nn.Module, roi_size: Tuple[int, int, int], overlap: float = 0.2) -> torch.Tensor:
    """Naive sliding-window inference without MONAI's internal indexing.
    Args:
        x: [B,C,D,H,W] tensor (typically B=C=1)
        model: nn.Module
        roi_size: (pd,ph,pw)
        overlap: float in [0,1)
    Returns:
        out tensor matching x shape on the same device.
    """
    model_was_train = model.training
    model.eval()
    with torch.no_grad():
        b, c, d, h, w = x.shape
        pd, ph, pw = roi_size
        sd = max(1, int(pd * (1 - overlap)))
        sh = max(1, int(ph * (1 - overlap)))
        sw = max(1, int(pw * (1 - overlap)))

        z_list = compute_grid(d, pd, sd)
        y_list = compute_grid(h, ph, sh)
        x_list = compute_grid(w, pw, sw)

        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for zz in z_list:
            for yy in y_list:
                for xx in x_list:
                    patch = x[:, :, zz:zz+pd, yy:yy+ph, xx:xx+pw]
                    pred = model(patch)
                    out[:, :, zz:zz+pd, yy:yy+ph, xx:xx+pw] += pred
                    cnt[:, :, zz:zz+pd, yy:yy+ph, xx:xx+pw] += 1.0

        out = out / torch.clamp(cnt, min=1e-6)
    if model_was_train:
        model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description="Train DynUNet for low-dose PET denoising")
    parser.add_argument("--drf", type=int, choices=[2, 4, 10, 20, 50, 100], default=10, help="Dose reduction factor (input folder '1-<DRF> dose')")
    args = parser.parse_args()

    # Update run identifier to include DRF, and derived artifact paths
    global RUN_ID, CURVE_PNG
    RUN_ID = f"{RUN_ID_BASE}_drf{args.drf}"
    CURVE_PNG = os.path.join(FIG_DIR, f"training_curves_{RUN_ID}.png")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_pids = find_patients(DATA_ROOT)
    if MAX_PATIENTS is not None:
        all_pids = all_pids[:MAX_PATIENTS]
    if not all_pids:
        print(f"No patient folders found in {DATA_ROOT}")
        return

    # Split train/val
    random.Random(RANDOM_SEED).shuffle(all_pids)
    n_val = max(1, int(len(all_pids) * VAL_FRACTION))
    val_pids = sorted(all_pids[:n_val])
    train_pids = sorted(all_pids[n_val:])
    if not train_pids:  # ensure at least one in train
        train_pids = val_pids
        val_pids = []

    print(f"Patients: train={len(train_pids)} val={len(val_pids)} total={len(all_pids)}")

    # Load volumes
    train_patients: List[PatientItem] = []
    val_patients: List[PatientItem] = []
    for pid in train_pids:
        try:
            train_patients.append(load_patient(DATA_ROOT, pid, args.drf))
        except Exception as e:
            print(f"Skipping train {pid}: {e}")
    for pid in val_pids:
        try:
            val_patients.append(load_patient(DATA_ROOT, pid, args.drf))
        except Exception as e:
            print(f"Skipping val {pid}: {e}")

    if not train_patients:
        print("No training patients loaded.")
        return

    # Build patch dataset
    train_ds = PatchDataset(train_patients, PATCH_SIZE, PATCH_STRIDE, augment=True)
    print(f"Train patches: {len(train_ds)}")

    # Train
    model, inferer, train_hist, val_hist = train_and_validate(train_ds, val_patients)

    # Save trained model with run id
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"dynunet_{RUN_ID}.pt")
    try:
        torch.save({
            'model_state': model.state_dict(),
            'run_id': RUN_ID,
            'config': {
                'patch_size': PATCH_SIZE,
                'patch_stride': PATCH_STRIDE,
                'epochs': EPOCHS,
                'lr': LR,
                'weight_decay': WEIGHT_DECAY,
                'loss_weights': (LOSS_MSE_W, LOSS_SSIM_W),
            }
        }, model_path)
        print(f"Saved model: {model_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

    # Save outputs for val
    save_all_outputs(model, inferer, val_patients)

    # Plot up to three validation examples (coronal slice grids)
    plot_val_examples(model, inferer, val_patients, n=3)

    print("Done.")


if __name__ == "__main__":
    main()
