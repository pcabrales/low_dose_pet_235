#!/usr/bin/env python3
"""
Inference script that matches train.py validation settings.

What it does
- Selects test NIfTI volumes of a chosen DRF from a CSV index.
- Uses the same robust normalization as train.py (0.1/99.9 percentiles).
- Runs MONAI's SlidingWindowInferer with the same ROI size as training
  and high overlap (0.75) as used in train.py validation.
- De-normalizes predictions back to the original intensity domain and saves
  the results. If the CSV indicates BQML units, outputs represent Bq/mL.
- Optionally renders a few example grids using train.py's plotter.

This script avoids any ad-hoc unit rescaling: it preserves the physical
units of the input NIfTI volume; when inputs are in BQML, outputs are in BQML.
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import json
import torch

import SimpleITK as sitk
from monai.inferers import SlidingWindowInferer

# Import helpers, constants, and plotting from training script
from train import (
    make_model,
    plot_val_examples,
    PatientItem,
    DEVICE,
    FIG_DIR,
    write_nifti,
    minmax_percentile_scale,
    PATCH_SIZE,
    # pretrained model paths and tuning per DRF
    PRETRAINED_MODEL_100DRF,
    NUM_LAYERS_100DRF,
    FILTERS_100DRF,
    PRETRAINED_MODEL_50DRF,
    NUM_LAYERS_50DRF,
    FILTERS_50DRF,
    PRETRAINED_MODEL_20DRF,
    NUM_LAYERS_20DRF,
    FILTERS_20DRF,
    PRETRAINED_MODEL_10DRF,
    NUM_LAYERS_10DRF,
    FILTERS_10DRF,
    PRETRAINED_MODEL_4DRF,
    NUM_LAYERS_4DRF,
    FILTERS_4DRF,
    PCT_LOW,
    PCT_HIGH,
    MODEL_NAME,
    GLOBAL_PCT_FILE,
)


DEFAULT_INPUT_DIR = "/root/PET_LOWDOSE/TEST_DATA/PET_Nifti/quadra"
DEFAULT_OUTPUT_DIR = "/root/PET_LOWDOSE/TEST_DATA/PET_Nifti/quadra-output"
INFO_CSV = "/root/PET_LOWDOSE/TEST_DATA/PET_info_noNORMAL.csv"


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

def write_nifti_with_units(volume: np.ndarray, meta: Dict, out_path: str, units: str = ""):
    """Write NIfTI with geometry from meta and annotate units in header 'descrip'."""
    img = sitk.GetImageFromArray(volume.astype(np.float32, copy=False))
    try:
        if meta and "spacing" in meta:
            img.SetSpacing(meta["spacing"])
        if meta and "origin" in meta:
            img.SetOrigin(meta["origin"])
        if meta and "direction" in meta:
            img.SetDirection(meta["direction"])
    except Exception:
        pass
    try:
        if units:
            # best-effort annotation for downstream consumers
            img.SetMetaData("descrip", f"Units={units}")
    except Exception:
        pass
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path)


def main():
    ap = argparse.ArgumentParser(description="Apply trained model to test NIfTI volumes")
    ap.add_argument("--drf", type=int, required=True, help="Dose reduction factor to select (e.g., 4, 10, 20, 50, 100)")
    ap.add_argument("--model", type=str, default="", help="Path to trained model .pt; if omitted, uses train.py's pretrained model mapping for the DRF or falls back to the newest trained-models/ match")
    ap.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--csv", type=str, default=INFO_CSV)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine model path and architecture consistent with train.py validation
    model_path = args.model
    num_layers = None
    filters = None
    if not model_path:
        # Prefer explicit pretrained mapping from train.py
        if args.drf == 100:
            model_path = PRETRAINED_MODEL_100DRF
            num_layers = NUM_LAYERS_100DRF
            filters = FILTERS_100DRF
        elif args.drf == 50:
            model_path = PRETRAINED_MODEL_50DRF
            num_layers = NUM_LAYERS_50DRF
            filters = FILTERS_50DRF
        elif args.drf == 20:
            model_path = PRETRAINED_MODEL_20DRF
            num_layers = NUM_LAYERS_20DRF
            filters = FILTERS_20DRF
        elif args.drf == 10:
            model_path = PRETRAINED_MODEL_10DRF
            num_layers = NUM_LAYERS_10DRF
            filters = FILTERS_10DRF
        elif args.drf == 4:
            model_path = PRETRAINED_MODEL_4DRF
            num_layers = NUM_LAYERS_4DRF
            filters = FILTERS_4DRF
        else:
            model_path = ""
        # If mapping missing/empty, fall back to newest match in trained-models
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
        raise RuntimeError("No model specified and no matching pretrained model found for the requested DRF.")

    # Load model
    model = make_model(model_name=MODEL_NAME, img_size=PATCH_SIZE, num_layers=(num_layers or 5), filters=filters).to(DEVICE)
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
            raise RuntimeError(f"Invalid DRF value in CSV for file {name}")
        if drf_row != args.drf:
            continue
        files.append((name, row))

    if not files:
        print(f"No NIfTI files found in {args.input_dir} for DRF={args.drf}.")
        return

    # MONAI sliding window inferer matching train.py validation overlap
    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian")  ###0.75

    # Load global normalization percentiles computed at training time
    try:
        with open(GLOBAL_PCT_FILE, "r") as f:
            g = json.load(f)
        p1_global = float(g["p_low"])
        p99_global = float(g["p_high"])
    except Exception:
        raise RuntimeError(f"Global percentile file not found or invalid: {GLOBAL_PCT_FILE}")

    # Predict each file
    vis_patients: List[PatientItem] = []
    for (fname, row) in files:
        in_path = os.path.join(args.input_dir, fname)
        vol, meta = read_nifti(in_path)  # [D,H,W]
        # Use global robust percentiles consistent with training normalization
        p1 = p1_global
        p99 = p99_global
        scale = float(max(1e-6, p99 - p1))
        
        # print(f"Global p999: {p99_global}, p1: {p1_global}")
        # print(f"Patient p99: {np.percentile(vol, 99.9)}, p1: {np.percentile(vol, 0.1)}")
    
        # Min–max normalize to [0,1]
        vol_mm = minmax_percentile_scale(vol, p1, p99)

        # Sliding-window prediction using MONAI inferer
        with torch.no_grad():
            xt = torch.from_numpy(vol_mm).float()[None, None].to(DEVICE)
            pred_mm = inferer(xt, model)
            pred_mm = pred_mm.clamp(0.0, None)
            pred_mm = pred_mm[0, 0].float().cpu().numpy()
        # De-normalize back to the input intensity domain
        pred = pred_mm * scale + p1
        # print(f"Total dose input: {vol.sum():.2f} Bq") # *meta.get('spacing', (1,1,1))[0]*meta.get('spacing', (1,1,1))[1]*meta.get('spacing', (1,1,1))[2]
        # print(f"Output dose sum: {pred.sum():.2f} Bq")
        # import matplotlib.pyplot as plt
        # NBINS = 200
        # counts, edges = np.histogram(vol[(vol > p1) & (vol < p99)], bins=NBINS)
        # counts_out, edges_out = np.histogram(pred[(pred > p1) & (pred < p99)], bins=NBINS)
        # centers_out = 0.5*(edges_out[1:] + edges_out[:-1])
        # centers = 0.5*(edges[1:] + edges[:-1])
        # plt.figure()
        # plt.step(centers, counts, where="mid", label="Input", color="blue", alpha=0.7)
        # plt.step(centers_out, counts_out, where="mid", label="Output", color="red", alpha=0.7)
        # plt.legend()
        # plt.title(f"Histogram of volume {fname} (DRF={args.drf})")
        # plt.xlabel("Intensity (Bq/mL)")
        # plt.ylabel("Voxel counts")
        # plt.grid(True)
        # plt.savefig(os.path.join("/root/low_dose_pet_235/figures", f"{os.path.splitext(fname)[0]}_hist_input.png"))
        # print(f"Input sum: {vol.sum():.2f} Bq")
        
        # Determine units from CSV; preserve BQML if already present
        units_val = (row.get("ImageUnits") or row.get("Units") or "").strip().upper()
        units_norm = "BQML" if ("BQML" in units_val or "BQ/ML" in units_val.replace(" ", "")) else units_val

        # Save output NIfTI; annotate units in header description best-effort
        out_path = os.path.join(args.output_dir, fname)
        try:
            # write geometry + annotate units
            write_nifti_with_units(pred, meta, out_path, units=units_norm)
        except Exception:
            print("Warning: failed to preserve geometry metadata; writing without it.")
            write_nifti_with_units(pred, {}, out_path, units=units_norm)
        print(f"Saved: {out_path} (units: {units_norm or 'unknown'})")

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
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_val_examples(model, inferer, vis_patients, n=3)


if __name__ == "__main__":
    main()
