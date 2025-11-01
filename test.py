#!/usr/bin/env python3
"""
Apply a trained DynUNet model to NIfTI files for a given DRF.

Inputs:
- Reads NIfTI volumes from /root/PET_LOWDOSE/TEST_DATA/PET_Nifti/quadra
- Uses PET_INFO CSV at /root/PET_LOWDOSE/TEST_DATA/PET_info_noNORMAL.csv
  - Second column: DoseReductionFactor (DRF)
  - Third column: NiftiFileName
  - Column RadionuclideTotalDose_Bq used to keep units consistent (BQML)

Behavior:
- Picks only files whose DRF matches --drf
- Normalizes per subject using robust min–max: map [p1, p99] -> [0,1],
  runs sliding-window inference with 96x96x96 patches and stride 48 (50% overlap), averages overlaps.
- De-normalizes back to original domain via p1 + out*(p99-p1), then keeps
  final values in Bq/ml (BQML). If CSV ImageUnits is already BQML, no additional scaling is applied.
- Saves outputs to /root/PET_LOWDOSE/TEST_DATA/PET_Nifti/quadra-output
  with the same filename as input.
- Also saves 3 visualization grids via the plotting helper in train_dynunet.py.

Minimal error handling by design.
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import SimpleITK as sitk
except Exception as e:
    print("SimpleITK is required.", file=sys.stderr)
    raise

try:
    from monai.inferers import SlidingWindowInferer
except Exception as e:
    print("MONAI is required.", file=sys.stderr)
    raise

# Import helpers and model definition from training script
from train_dynunet import (
    make_model,
    compute_grid,
    enumerate_patches,
    plot_val_examples,
    PatientItem,
    DEVICE,
    FIG_DIR,
    write_nifti,
    minmax_percentile_scale,
)


DEFAULT_INPUT_DIR = "/root/PET_LOWDOSE/TEST_DATA/PET_Nifti/quadra"
DEFAULT_OUTPUT_DIR = "/root/PET_LOWDOSE/TEST_DATA/PET_Nifti/quadra-output"
INFO_CSV = "/root/PET_LOWDOSE/TEST_DATA/PET_info_noNORMAL.csv"

PATCH_SIZE = (96, 96, 96)
STRIDE = (48, 48, 48)


def load_csv_index(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Index CSV rows by NiftiFileName (basename)."""
    idx: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("NiftiFileName") or "").strip()
            if not name:
                continue
            idx[os.path.basename(name)] = row
    return idx


def read_nifti(path: str) -> Tuple[np.ndarray, Dict]:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)  # [D,H,W]
    meta = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
    }
    return arr, meta


def sliding_window_predict(vol_mm: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """Predict full volume with 50% overlap averaging in [0,1] space.

    Args:
        vol_mm: min–max normalized volume [D,H,W]
        model: torch model (eval mode)
    Returns:
        predicted normalized volume [D,H,W]
    """
    d, h, w = vol_mm.shape
    pd, ph, pw = PATCH_SIZE
    sd, sh, sw = STRIDE

    coords = enumerate_patches((d, h, w), PATCH_SIZE, STRIDE)

    out = np.zeros((d, h, w), dtype=np.float32)
    cnt = np.zeros((d, h, w), dtype=np.float32)

    with torch.no_grad():
        for (z, y, x) in coords:
            patch = vol_mm[z:z+pd, y:y+ph, x:x+pw]
            xt = torch.from_numpy(patch)[None, None].to(DEVICE)
            pred = model(xt)
            pnp = pred[0, 0].float().detach().cpu().numpy()
            out[z:z+pd, y:y+ph, x:x+pw] += pnp
            cnt[z:z+pd, y:y+ph, x:x+pw] += 1.0

    cnt = np.clip(cnt, 1e-6, None)
    out /= cnt
    return out


def main():
    ap = argparse.ArgumentParser(description="Apply trained DynUNet to test NIfTI volumes")
    ap.add_argument("--drf", type=int, required=True, help="Dose reduction factor to select (e.g., 4, 10, 20, 50, 100)")
    ap.add_argument("--model", type=str, default="", help="Path to trained model .pt; if omitted, picks latest matching drf from trained-models/")
    ap.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--csv", type=str, default=INFO_CSV)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find model path
    model_path = args.model
    if not model_path:
        tm_dir = os.path.join(os.path.dirname(__file__), "trained-models")
        cand = []
        if os.path.isdir(tm_dir):
            for n in os.listdir(tm_dir):
                if n.endswith(".pt") and f"drf{args.drf}" in n:
                    cand.append(os.path.join(tm_dir, n))
        if cand:
            cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            model_path = cand[0]
    if not model_path:
        print("No model specified and no matching trained-models/ file found.")
        return

    # Load model
    model = make_model().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()

    # Build CSV index
    info = load_csv_index(args.csv)

    # Gather NIfTI files of this DRF
    files = []
    for name in sorted(os.listdir(args.input_dir)):
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            continue
        row = info.get(name)
        if not row:
            continue
        try:
            drf_row = int(float(row.get("DoseReductionFactor", "")))
        except Exception:
            continue
        if drf_row != args.drf:
            continue
        files.append((name, row))

    if not files:
        print(f"No NIfTI files found in {args.input_dir} for DRF={args.drf}.")
        return

    # Prepare MONAI inferer for plotting helper only (uses its forward internally)
    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian")

    # Predict each file
    vis_patients: List[PatientItem] = []
    for (fname, row) in files:
        in_path = os.path.join(args.input_dir, fname)
        vol, meta = read_nifti(in_path)  # [D,H,W]

        # Robust percentiles per subject
        p1 = float(np.percentile(vol, 1.0))
        p99 = float(np.percentile(vol, 99.0))
        scale = float(max(1e-6, p99 - p1))

        # Min–max normalize
        vol_mm = minmax_percentile_scale(vol, p1, p99)

        # Sliding-window prediction in [0,1]
        pred_mm = sliding_window_predict(vol_mm, model)

        # De-normalize back to original domain
        pred = pred_mm * scale + p1

        # Keep Bq/ml (BQML) units using CSV (no-op if already BQML)
        units = (row.get("ImageUnits") or "").strip().upper()
        try:
            dose_bq = float(row.get("RadionuclideTotalDose_Bq", 1.0))
        except Exception:
            dose_bq = 1.0
        if units == "BQML":
            scale = 1.0
        else:
            # Simple fallback if units differ: scale by dose (best-effort minimal behavior)
            scale = float(dose_bq) if dose_bq > 0 else 1.0
        pred_bqml = pred * scale

        # Save output NIfTI
        out_path = os.path.join(args.output_dir, fname)
        try:
            write_nifti(pred_bqml, meta, out_path)
        except Exception:
            write_nifti(pred_bqml, {}, out_path)
        print(f"Saved: {out_path}")

        # Collect up to 3 cases for visualization. Use input as 'target' to view diffs.
        if len(vis_patients) < 3:
            # Re-create arrays expected by plotter: min–max normalized
            pitem = PatientItem(
                pid=os.path.splitext(os.path.splitext(fname)[0])[0],
                input_arr=vol_mm.astype(np.float32, copy=False),
                target_arr=vol_mm.astype(np.float32, copy=False),  # use input as pseudo-target
                input_meta=meta,
                target_meta=meta,
            )
            vis_patients.append(pitem)

    # Save visualizations for up to 3 samples
    try:
        os.makedirs(FIG_DIR, exist_ok=True)
        plot_val_examples(model, inferer, vis_patients, n=3)
    except Exception:
        pass


if __name__ == "__main__":
    main()
