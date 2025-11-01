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
import re
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

import numpy as np
try:
    import zarr
    from numcodecs import Blosc
    try:
        from zarr.storage import DirectoryStore as _DirectoryStore
    except Exception:
        _DirectoryStore = None
except Exception as e:
    print("Zarr and numcodecs are required for Zarr-based IO.", file=sys.stderr)
    raise
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
# Zarr arrays live under DATA_ROOT/zarr as `<pid>__drf{drf}.zarr` and `<pid>__full.zarr`
ZARR_ROOT = os.path.join(DATA_ROOT, "zarr")
NPY_ROOT = "/dev/null/unused_npz_path"  # legacy (unused after Zarr migration)
META_CSV = "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/all_subjects/metadata.csv"
OUTPUT_ROOT = "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output"

# Local artifact directories
BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "trained-models")
RUN_ID = time.strftime("%Y%m%d_%H%M")  # updated later to include DRF
RUN_ID_BASE = RUN_ID

# Default to DynUNet per README; nnFormer is much heavier
MODEL_NAME = "dynunet"  ### choices: "DynUNet" or "nnFormer"
PATCH_SIZE = (128, 128, 128)  ###(96, 96, 96)  ###(80, 80, 80)
# Slight overlap: stride < patch size; 80 gives 16 voxels overlap
PATCH_STRIDE = (96, 96, 96)  ###(80, 80, 80)  ###(64, 64, 64)

MAX_PATIENTS = 10  ###None  ### set to an int to limit for quick tests
VAL_FRACTION = 0.05  
RANDOM_SEED = 42

BATCH_SIZE = 16  ###
EPOCHS = 3  ###15  ###
LR = 1e-4
WEIGHT_DECAY = 1e-5

LOSS_MSE_W = 0.9
LOSS_SSIM_W = 0.1

CURVE_PNG = os.path.join(FIG_DIR, f"training_curves_{RUN_ID}.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Enable matmul acceleration on Ampere+ GPUs (helps nnFormer/attention)
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# Use tuned convolution algorithms when input shapes are static
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except Exception:
    pass


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


# -------------------- DICOM -> ZARR caching utils ----------------------

def _serialize_vec(vec) -> str:
    try:
        return "|".join([f"{float(v):.8g}" for v in list(vec)])
    except Exception:
        return ""


def _parse_vec(s: str) -> Optional[Tuple[float, ...]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split("|") if p.strip() != ""]
    try:
        return tuple(float(p) for p in parts)
    except Exception:
        return None


def _parse_shape(s: str) -> Optional[Tuple[int, int, int]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split("|") if p.strip() != ""]
    if len(parts) != 3:
        return None
    try:
        d, h, w = (int(float(p)) for p in parts)
        return (d, h, w)
    except Exception:
        return None


def _series_match_drf(name: str) -> Optional[int]:
    """Return DRF int if folder looks like '1-10 dose', else None."""
    m = re.match(r"^1-(\d+)\s*dose$", name.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _zarr_write_array(zpath: str, arr: np.ndarray, meta: Dict):
    """Create a Zarr array at zpath with chunks tuned for 3D patches and Blosc compression.

    Stores spacing/origin/direction in attrs best-effort.
    """
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    # Choose chunking to align reasonably with training patches to reduce IO
    d, h, w = arr.shape
    cz = min(32, d)
    cy = min(96, h)
    cx = min(96, w)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    # Prefer Zarr v2 store if available; otherwise use path-based open
    try:
        if _DirectoryStore is not None:
            store = _DirectoryStore(zpath)
            z = zarr.create(store=store, shape=arr.shape, chunks=(cz, cy, cx), dtype="f4",
                            compressor=compressor, overwrite=True, zarr_format=2)
        else:
            z = zarr.open(zpath, mode="w", shape=arr.shape, chunks=(cz, cy, cx), dtype="f4",
                          compressor=compressor, zarr_format=2)
    except TypeError:
        # Signature differences or zarr_format not supported. Try without zarr_format (v2 default).
        try:
            if _DirectoryStore is not None:
                store = _DirectoryStore(zpath)
                z = zarr.create(store=store, shape=arr.shape, chunks=(cz, cy, cx), dtype="f4",
                                compressor=compressor, overwrite=True)
            else:
                z = zarr.open(zpath, mode="w", shape=arr.shape, chunks=(cz, cy, cx), dtype="f4",
                              compressor=compressor)
        except Exception:
            # Last resort: write uncompressed array (works with v3 defaults)
            z = zarr.open(zpath, mode="w", shape=arr.shape, chunks=(cz, cy, cx), dtype="f4")
    z[:] = arr.astype(np.float32, copy=False)
    # Save minimal metadata
    try:
        if hasattr(z, "attrs"):
            if meta:
                for k in ("spacing", "origin", "direction"):
                    v = meta.get(k)
                    if v is not None:
                        # store as list for JSON-compat
                        try:
                            z.attrs[k] = list(v)
                        except Exception:
                            pass
    except Exception:
        pass


def convert_all_to_zarr_if_needed(data_root: str = DATA_ROOT, zarr_root: str = ZARR_ROOT, meta_csv_path: str = META_CSV):
    """Convert all .IMA series to Zarr arrays if not already present.

    For each patient it writes:
      - `<pid>__full.zarr` for target (Full_dose)
      - `<pid>__drf{drf}.zarr` for each input series found (folders like `1-<drf> dose`)

    Notes:
      - Does not rewrite metadata.csv if it already exists (per user request).
      - If ZARR_ROOT contains arrays, conversion is skipped for existing ones.
    """
    os.makedirs(zarr_root, exist_ok=True)

    pids = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    print(f"Ensuring Zarr cache for {len(pids)} patients at {zarr_root} ...")
    for pid in tqdm(pids, desc="Zarr", unit="pid"):
        pdir = os.path.join(data_root, pid)
        if not os.path.isdir(pdir):
            continue

        # 1) Target series (Full_dose)
        try:
            tgt_dir = os.path.join(pdir, "Full_dose")
            zpath = os.path.join(zarr_root, f"{pid}__full.zarr")
            exists_ok = False
            if os.path.isdir(zpath):
                # verify openable
                try:
                    _ = zarr.open(zpath, mode="r")
                    exists_ok = True
                except Exception:
                    exists_ok = False
            if not exists_ok and os.path.isdir(tgt_dir):
                vol, meta = read_dicom_series(tgt_dir)
                if vol.dtype == np.float64:
                    vol = vol.astype(np.float32, copy=False)
                _zarr_write_array(zpath, vol, meta)
        except Exception as e:
            print(f"Warning: Zarr convert target failed for {pid}: {e}")

        # 2) Input series (any folder matching '1-<drf> dose')
        try:
            for sub in sorted(os.listdir(pdir)):
                drf = _series_match_drf(sub)
                if drf is None:
                    continue
                sdir = os.path.join(pdir, sub)
                zpath = os.path.join(zarr_root, f"{pid}__drf{drf}.zarr")
                exists_ok = False
                if os.path.isdir(zpath):
                    try:
                        _ = zarr.open(zpath, mode="r")
                        exists_ok = True
                    except Exception:
                        exists_ok = False
                if exists_ok:
                    continue
                try:
                    vol, meta = read_dicom_series(sdir)
                    if vol.dtype == np.float64:
                        vol = vol.astype(np.float32, copy=False)
                    _zarr_write_array(zpath, vol, meta)
                except Exception as e:
                    print(f"Warning: Zarr convert input failed {pid} {sub}: {e}")
        except Exception:
            pass

    # Respect user request: do not rewrite metadata.csv if it already exists.
    if not os.path.isfile(meta_csv_path):
        # Best-effort minimal metadata if missing: write pid/series/drf/shape only.
        try:
            rows: List[Dict[str, str]] = []
            for pid in pids:
                # target
                tz = os.path.join(zarr_root, f"{pid}__full.zarr")
                if os.path.isdir(tz):
                    try:
                        za = zarr.open(tz, mode="r")
                        s = za.shape
                        rows.append({
                            "pid": pid, "series": "target", "drf": "full",
                            "file": f"{pid}__full.zarr",  # for reference
                            "shape": f"{s[0]}|{s[1]}|{s[2]}",
                            "spacing": "", "origin": "", "direction": "",
                            "p1": "", "p99": "",
                        })
                    except Exception:
                        pass
                # inputs
                pdir = os.path.join(data_root, pid)
                try:
                    for sub in sorted(os.listdir(pdir)):
                        drf = _series_match_drf(sub)
                        if drf is None:
                            continue
                        iz = os.path.join(zarr_root, f"{pid}__drf{drf}.zarr")
                        if not os.path.isdir(iz):
                            continue
                        try:
                            za = zarr.open(iz, mode="r")
                            s = za.shape
                            rows.append({
                                "pid": pid, "series": "input", "drf": str(drf),
                                "file": f"{pid}__drf{drf}.zarr",
                                "shape": f"{s[0]}|{s[1]}|{s[2]}",
                                "spacing": "", "origin": "", "direction": "",
                                "p1": "", "p99": "",
                            })
                        except Exception:
                            pass
                except Exception:
                    pass
            if rows:
                with open(meta_csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "pid", "series", "drf", "file", "shape", "spacing", "origin", "direction", "p1", "p99"
                    ])
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                print(f"Wrote minimal metadata: {meta_csv_path}")
        except Exception as e:
            print(f"Failed to write minimal metadata CSV: {e}")


# --------------------------- Data Handling -----------------------------

@dataclass
class PatientItem:
    pid: str
    input_arr: np.ndarray  # [D,H,W]
    target_arr: np.ndarray  # [D,H,W]
    input_meta: Dict
    target_meta: Dict


@dataclass
class NpyEntry:
    """Lightweight descriptor for a patient pair stored as NPY files."""
    pid: str
    in_path: str
    tg_path: str
    pair_shape: Tuple[int, int, int]  # min dims (D,H,W) across input/target
    in_meta: Dict
    tg_meta: Dict


@dataclass
class ZarrEntry:
    """Descriptor for a patient pair stored as Zarr arrays."""
    pid: str
    in_path: str
    tg_path: str
    pair_shape: Tuple[int, int, int]
    in_meta: Dict
    tg_meta: Dict


def minmax_percentile_scale(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Map [p1, p99] to [0, 1] and clamp outside.

    Assumes p99 > p1; falls back to zeros if not.
    """
    scale = float(p99 - p1)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x.astype(np.float32) - float(p1)) / scale
    return np.clip(y, 0.0, 1.0, out=y)


def find_patients(data_root: str) -> List[str]:
    pids = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if os.path.isdir(p):
            pids.append(name)
    return pids


def load_metadata_index(meta_csv_path: str = META_CSV) -> Dict[Tuple[str, str, str], Dict]:
    """Load metadata CSV into a dict keyed by (series,input/target) and (pid,drf).

    Key format: (series, pid, drf_str) where drf_str is str(int) for inputs and 'full' for target.
    Values contain keys: file, spacing, origin, direction, shape.
    """
    idx: Dict[Tuple[str, str, str], Dict] = {}
    if not os.path.isfile(meta_csv_path):
        return idx
    try:
        with open(meta_csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                key = (row.get("series", ""), row.get("pid", ""), row.get("drf", ""))
                if not key[0] or not key[1] or not key[2]:
                    continue
                idx[key] = {
                    "file": row.get("file", ""),
                    "shape": row.get("shape", ""),
                    "spacing": _parse_vec(row.get("spacing", "")),
                    "origin": _parse_vec(row.get("origin", "")),
                    "direction": _parse_vec(row.get("direction", "")),
                    "p1": _parse_float(row.get("p1", "")),
                    "p99": _parse_float(row.get("p99", "")),
                }
    except Exception as e:
        print(f"Warning: failed to read metadata.csv: {e}")
    return idx


def build_npy_entries(meta_idx: Dict[Tuple[str, str, str], Dict], pids: List[str], drf: int, npy_root: str = NPY_ROOT) -> List[NpyEntry]:
    entries: List[NpyEntry] = []
    for pid in pids:
        in_key = ("input", pid, str(drf))
        tg_key = ("target", pid, "full")
        if in_key not in meta_idx or tg_key not in meta_idx:
            continue
        in_rec = meta_idx[in_key]
        tg_rec = meta_idx[tg_key]
        in_shape = _parse_shape(in_rec.get("shape", ""))
        tg_shape = _parse_shape(tg_rec.get("shape", ""))
        if not in_shape or not tg_shape:
            continue
        pair_shape = (min(in_shape[0], tg_shape[0]), min(in_shape[1], tg_shape[1]), min(in_shape[2], tg_shape[2]))
        entries.append(NpyEntry(
            pid=pid,
            in_path=os.path.join(npy_root, in_rec["file"]),
            tg_path=os.path.join(npy_root, tg_rec["file"]),
            pair_shape=pair_shape,
            in_meta={
                "spacing": in_rec.get("spacing"),
                "origin": in_rec.get("origin"),
                "direction": in_rec.get("direction"),
                "p1": in_rec.get("p1"),
                "p99": in_rec.get("p99"),
            },
            tg_meta={
                "spacing": tg_rec.get("spacing"),
                "origin": tg_rec.get("origin"),
                "direction": tg_rec.get("direction"),
                "p1": tg_rec.get("p1"),
                "p99": tg_rec.get("p99"),
            },
        ))
    return entries


def build_zarr_entries(meta_idx: Dict[Tuple[str, str, str], Dict], pids: List[str], drf: int, zarr_root: str = ZARR_ROOT) -> List[ZarrEntry]:
    entries: List[ZarrEntry] = []
    for pid in pids:
        in_key = ("input", pid, str(drf))
        tg_key = ("target", pid, "full")
        if in_key not in meta_idx or tg_key not in meta_idx:
            # still allow if Zarr files exist; infer shape from arrays
            in_z = os.path.join(zarr_root, f"{pid}__drf{drf}.zarr")
            tg_z = os.path.join(zarr_root, f"{pid}__full.zarr")
            if os.path.isdir(in_z) and os.path.isdir(tg_z):
                try:
                    zi = zarr.open(in_z, mode="r"); zt = zarr.open(tg_z, mode="r")
                    pair_shape = (
                        min(zi.shape[0], zt.shape[0]),
                        min(zi.shape[1], zt.shape[1]),
                        min(zi.shape[2], zt.shape[2]),
                    )
                    entries.append(ZarrEntry(
                        pid=pid,
                        in_path=in_z,
                        tg_path=tg_z,
                        pair_shape=pair_shape,
                        in_meta={},
                        tg_meta={},
                    ))
                except Exception:
                    pass
            continue
        in_rec = meta_idx[in_key]
        tg_rec = meta_idx[tg_key]
        in_shape = _parse_shape(in_rec.get("shape", ""))
        tg_shape = _parse_shape(tg_rec.get("shape", ""))
        if not in_shape or not tg_shape:
            # Fallback: probe zarr
            in_z = os.path.join(zarr_root, f"{pid}__drf{drf}.zarr")
            tg_z = os.path.join(zarr_root, f"{pid}__full.zarr")
            try:
                zi = zarr.open(in_z, mode="r"); zt = zarr.open(tg_z, mode="r")
                in_shape = zi.shape; tg_shape = zt.shape
            except Exception:
                continue
        pair_shape = (min(in_shape[0], tg_shape[0]), min(in_shape[1], tg_shape[1]), min(in_shape[2], tg_shape[2]))
        in_z = os.path.join(zarr_root, f"{pid}__drf{drf}.zarr")
        tg_z = os.path.join(zarr_root, f"{pid}__full.zarr")
        if not (os.path.isdir(in_z) and os.path.isdir(tg_z)):
            continue
        entries.append(ZarrEntry(
            pid=pid,
            in_path=in_z,
            tg_path=tg_z,
            pair_shape=pair_shape,
            in_meta={
                "spacing": in_rec.get("spacing"),
                "origin": in_rec.get("origin"),
                "direction": in_rec.get("direction"),
                "p1": in_rec.get("p1"),
                "p99": in_rec.get("p99"),
            },
            tg_meta={
                "spacing": tg_rec.get("spacing"),
                "origin": tg_rec.get("origin"),
                "direction": tg_rec.get("direction"),
                "p1": tg_rec.get("p1"),
                "p99": tg_rec.get("p99"),
            },
        ))
    return entries


def load_patient_from_npy(npy_root: str, meta_idx: Dict[Tuple[str, str, str], Dict], pid: str, drf: int) -> PatientItem:
    """Load a patient input/target pair from NPY cache, crop-align, and min-max scale via subject-level robust percentiles."""
    in_key = ("input", pid, str(drf))
    tg_key = ("target", pid, "full")
    if in_key not in meta_idx or tg_key not in meta_idx:
        raise RuntimeError(f"Missing NPY/meta for pid={pid} drf={drf}")

    in_rec = meta_idx[in_key]
    tg_rec = meta_idx[tg_key]
    in_path = os.path.join(npy_root, in_rec["file"])
    tg_path = os.path.join(npy_root, tg_rec["file"])
    if not os.path.isfile(in_path) or not os.path.isfile(tg_path):
        raise RuntimeError(f"NPY files not found for pid={pid} drf={drf}")

    x = np.load(in_path)
    in_vol = x["arr"] if isinstance(x, np.lib.npyio.NpzFile) else x
    x = np.load(tg_path)
    tgt_vol = x["arr"] if isinstance(x, np.lib.npyio.NpzFile) else x

    # Crop-align to min dims
    dz = min(in_vol.shape[0], tgt_vol.shape[0])
    dy = min(in_vol.shape[1], tgt_vol.shape[1])
    dx = min(in_vol.shape[2], tgt_vol.shape[2])
    in_vol = in_vol[:dz, :dy, :dx]
    tgt_vol = tgt_vol[:dz, :dy, :dx]

    # Subject-level percentiles: prefer target's p1/p99 if available, else input's
    p1 = tg_rec.get("p1") if (tg_rec.get("p1") is not None) else in_rec.get("p1")
    p99 = tg_rec.get("p99") if (tg_rec.get("p99") is not None) else in_rec.get("p99")
    if p1 is None or p99 is None:
        # Fallback compute on the fly
        try:
            p1 = float(np.percentile(tgt_vol, 1.0))
            p99 = float(np.percentile(tgt_vol, 99.0))
        except Exception:
            p1, p99 = 0.0, 1.0
    in_vol = minmax_percentile_scale(in_vol, float(p1), float(p99)).astype(np.float32, copy=False)
    tgt_vol = minmax_percentile_scale(tgt_vol, float(p1), float(p99)).astype(np.float32, copy=False)

    in_meta = {
        "spacing": in_rec.get("spacing"),
        "origin": in_rec.get("origin"),
        "direction": in_rec.get("direction"),
    }
    tg_meta = {
        "spacing": tg_rec.get("spacing"),
        "origin": tg_rec.get("origin"),
        "direction": tg_rec.get("direction"),
    }
    return PatientItem(pid=pid, input_arr=in_vol, target_arr=tgt_vol, input_meta=in_meta, target_meta=tg_meta)


def load_patient_from_zarr(zarr_root: str, meta_idx: Dict[Tuple[str, str, str], Dict], pid: str, drf: int) -> PatientItem:
    """Load a patient input/target pair from Zarr, crop-align, and min-max scale via subject-level robust percentiles."""
    in_path = os.path.join(zarr_root, f"{pid}__drf{drf}.zarr")
    tg_path = os.path.join(zarr_root, f"{pid}__full.zarr")
    if not (os.path.isdir(in_path) and os.path.isdir(tg_path)):
        raise RuntimeError(f"Zarr files not found for pid={pid} drf={drf}")

    xi = zarr.open(in_path, mode="r")
    yi = zarr.open(tg_path, mode="r")
    in_vol = np.asarray(xi[:])
    tgt_vol = np.asarray(yi[:])

    # Crop-align to min dims
    dz = min(in_vol.shape[0], tgt_vol.shape[0])
    dy = min(in_vol.shape[1], tgt_vol.shape[1])
    dx = min(in_vol.shape[2], tgt_vol.shape[2])
    in_vol = in_vol[:dz, :dy, :dx]
    tgt_vol = tgt_vol[:dz, :dy, :dx]

    # Metadata percentiles if available
    in_key = ("input", pid, str(drf))
    tg_key = ("target", pid, "full")
    in_rec = meta_idx.get(in_key, {})
    tg_rec = meta_idx.get(tg_key, {})
    p1 = tg_rec.get("p1") if (tg_rec.get("p1") is not None) else in_rec.get("p1")
    p99 = tg_rec.get("p99") if (tg_rec.get("p99") is not None) else in_rec.get("p99")
    if p1 is None or p99 is None:
        try:
            p1 = float(np.percentile(tgt_vol, 1.0))
            p99 = float(np.percentile(tgt_vol, 99.0))
        except Exception:
            p1, p99 = 0.0, 1.0
    in_vol = minmax_percentile_scale(in_vol, float(p1), float(p99)).astype(np.float32, copy=False)
    tgt_vol = minmax_percentile_scale(tgt_vol, float(p1), float(p99)).astype(np.float32, copy=False)

    in_meta = {
        "spacing": in_rec.get("spacing"),
        "origin": in_rec.get("origin"),
        "direction": in_rec.get("direction"),
    }
    tg_meta = {
        "spacing": tg_rec.get("spacing"),
        "origin": tg_rec.get("origin"),
        "direction": tg_rec.get("direction"),
    }
    return PatientItem(pid=pid, input_arr=in_vol, target_arr=tgt_vol, input_meta=in_meta, target_meta=tg_meta)


def load_patient(data_root: str, pid: str, drf: int) -> PatientItem:
    """Deprecated DICOM loader retained for reference; now loads from Zarr cache."""
    meta_idx = load_metadata_index(META_CSV)
    return load_patient_from_zarr(ZARR_ROOT, meta_idx, pid, drf)


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


class LazyPatchDataset(Dataset):
    """Patch dataset that loads from disk lazily using Zarr arrays or legacy NPY.

    For Zarr, it opens arrays once per-worker and slices patches, letting Zarr handle
    chunked IO. For legacy .npz, it builds a memmap side-cache (.npy) for faster access.
    Normalizes using subject-level robust percentiles p1/p99 from metadata when available.
    """

    def __init__(self, entries: List[NpyEntry], patch_size: Tuple[int, int, int], stride: Tuple[int, int, int], augment: bool = True):
        self.entries = entries
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment

        # Precompute patch coordinates for all entries using pair_shape
        self.index: List[Tuple[int, Tuple[int, int, int]]] = []  # (entry_idx, (z,y,x))
        for ei, e in enumerate(self.entries):
            coords = enumerate_patches(e.pair_shape, patch_size, stride)
            for c in coords:
                self.index.append((ei, c))

        # Lightweight per-worker LRU cache for loaded arrays (Zarr or memmap-backed .npy)
        self._pair_cache: Dict[int, Tuple[object, object]] = {}
        self._pair_lru: List[int] = []
        # Keep more subjects in the per-worker cache to reduce IO thrash when shuffling
        self._pair_cap: int = 16  # LRU size per worker

    def __len__(self) -> int:
        return len(self.index)

    # No helper memoization needed for minimal prototype

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ei, (z, y, x) = self.index[i]
        e = self.entries[ei]
        pd, ph, pw = self.patch_size

        # Obtain arrays from cache (memmap-backed where possible)
        xi, yi = self._get_pair_arrays(ei, e)

        # Slice needed patch window
        xin = np.asarray(xi[z:z+pd, y:y+ph, x:x+pw])
        ygt = np.asarray(yi[z:z+pd, y:y+ph, x:x+pw])

        # Subject-level robust min-max using shared p1/p99 (prefer target metadata)
        p1 = e.tg_meta.get("p1") if (e.tg_meta.get("p1") is not None) else e.in_meta.get("p1")
        p99 = e.tg_meta.get("p99") if (e.tg_meta.get("p99") is not None) else e.in_meta.get("p99")
        p1 = float(p1 if p1 is not None else 0.0)
        p99 = float(p99 if p99 is not None else 1.0)
        xin = minmax_percentile_scale(xin, p1, p99)
        ygt = minmax_percentile_scale(ygt, p1, p99)

        xt = torch.from_numpy(xin).unsqueeze(0)
        yt = torch.from_numpy(ygt).unsqueeze(0)

        if self.augment:
            xt, yt = random_augment(xt, yt)

        # Return subject-level p1/p99 to support empty-patch skipping in the training loop.
        stats = torch.tensor([p1, p99], dtype=torch.float32)

        return xt, yt, stats

    def _get_pair_arrays(self, key: int, e: NpyEntry) -> Tuple[object, object]:
        # Fast path: in LRU cache
        if key in self._pair_cache:
            # update LRU order
            try:
                self._pair_lru.remove(key)
            except ValueError:
                pass
            self._pair_lru.append(key)
            return self._pair_cache[key]

        # Otherwise load arrays (Zarr or memmap-backed .npy)
        xi = self._load_array_with_cache(e.in_path)
        yi = self._load_array_with_cache(e.tg_path)

        # Compute and cache subject-level robust percentiles once per pair if missing
        try:
            if (e.tg_meta.get("p1") is None) or (e.tg_meta.get("p99") is None):
                src = yi if yi is not None else xi
                # src may be zarr array; ensure ndarray to compute percentiles
                if hasattr(src, "__getitem__") and not isinstance(src, np.ndarray):
                    arr = np.asarray(src[:])
                else:
                    arr = np.asarray(src)
                p1 = float(np.percentile(arr, 1.0))
                p99 = float(np.percentile(arr, 99.0))
                e.tg_meta["p1"], e.tg_meta["p99"] = p1, p99
        except Exception:
            pass

        # Insert into LRU and evict if needed
        self._pair_cache[key] = (xi, yi)
        self._pair_lru.append(key)
        if len(self._pair_lru) > self._pair_cap:
            old = self._pair_lru.pop(0)
            try:
                del self._pair_cache[old]
            except KeyError:
                pass
        return xi, yi

    def _load_array_with_cache(self, path: str):
        # If path is a Zarr directory, open with zarr
        if path.endswith('.zarr') and os.path.isdir(path):
            try:
                return zarr.open(path, mode='r')
            except Exception:
                pass
        # If path is .npy, open as memmap for zero-copy slices
        if path.endswith('.npy'):
            try:
                return np.load(path, mmap_mode='r')
            except Exception:
                # fallback: standard load
                x = np.load(path)
                return x

        # If path is .npz, create a sibling .npy cache once, then memmap it
        if path.endswith('.npz'):
            cache_path = path[:-4] + '__cache.npy'
            if not os.path.isfile(cache_path):
                # Atomic-ish write to avoid races across workers
                tmp_path = cache_path + f'.tmp_{os.getpid()}'
                try:
                    x = np.load(path)
                    arr = x['arr'] if isinstance(x, np.lib.npyio.NpzFile) else x
                    # ensure float32 for training
                    np.save(tmp_path, np.asarray(arr, dtype=np.float32, copy=False))
                    os.replace(tmp_path, cache_path)
                except Exception:
                    # If something went wrong, attempt to clean temp and fall back to direct load
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    x = np.load(path)
                    return x['arr'] if isinstance(x, np.lib.npyio.NpzFile) else x
            # Open memmap
            try:
                return np.load(cache_path, mmap_mode='r')
            except Exception:
                x = np.load(cache_path)
                return x
        # Unknown extension; fallback
        x = np.load(path)
        return x['arr'] if isinstance(x, np.lib.npyio.NpzFile) else x


# ----------------------------- Model -----------------------------------

def make_model(model_name: str = "DynUNet", img_size: Tuple[int, int, int] = (96, 96, 96)) -> nn.Module:
    if model_name.lower() == "dynunet":
        kernel_size = [[3, 3, 3]] * 5
        strides = [1, 2, 2, 2, 2]
        up_kernels = [2, 2, 2, 2]
        # add to make the model larger: filters=[16, 32, 64, 128, 256, 512]
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
    elif model_name.lower() == "nnformer":
        from nnFormer.nnFormer_seg import nnFormer
        model = nnFormer(crop_size=img_size,
                    embedding_dim=64,
                    input_channels=1,
                    num_classes=1,
                    depths=[2,2,2,2],
                    num_heads=[4, 8, 16, 32])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
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

    model = make_model(model_name=MODEL_NAME, img_size=PATCH_SIZE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    mse = nn.MSELoss()
    ssim = SSIMLoss(spatial_dims=3, data_range=1.0)  # min-max scaled to [0,1]

    # MONAI sliding window inferer for validation/inference
    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.2, mode="gaussian")

    # Keep worker count modest to avoid CPU thrash from IO/decompression
    # DataLoader tuning: more workers, pinned memory for faster H2D, and prefetching
    nw = max(2, min(8, (os.cpu_count() or 8)))
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=nw,
        pin_memory=(DEVICE.type == "cuda"),
        prefetch_factor=4,
        persistent_workers=True,
    )
    

    train_hist: List[float] = []
    val_hist: List[float] = []

    # Track and optionally print GPU memory usage once during training
    printed_gpu_mem = False
    if DEVICE.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # Ensure model directory exists for per-epoch checkpoints
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Mixed precision can significantly speed up training on GPUs
    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        skipped_empty = 0
        total_seen = 0
        t0 = time.time()
        for batch in tqdm(loader, desc="Training", unit="batch", leave=False):
            # Support datasets returning (x,y) or (x,y,stats)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, gstats = batch
            else:
                xb, yb = batch  # type: ignore
                gstats = None

            xb = xb.to(DEVICE, non_blocking=True)  # [B,C,D,H,W]
            yb = yb.to(DEVICE, non_blocking=True)

            # Skip empty patches, but keep some to ensure it works on inference
            with torch.no_grad():
                flat = xb.view(xb.shape[0], -1)
                # stats in [0,1] space
                p_max_z = flat.max(dim=1).values
                # Threshold: if maximum of normalized patch [0,1] <= thresh 1% and not kept due to randomness, skip
                thresh = 0.01
                prob_keep_empty = 0.3
                keep = (p_max_z > thresh) | (
                    (p_max_z <= thresh) & (torch.rand_like(p_max_z) < prob_keep_empty)
                )

            total_seen += xb.shape[0]
            if keep.sum().item() == 0:
                skipped_empty += xb.shape[0]
                continue
            if keep.sum().item() < xb.shape[0]:
                skipped_empty += int((~keep).sum().item())
                xb = xb[keep]
                yb = yb[keep]
                
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                l = LOSS_MSE_W * mse(pred, yb) + LOSS_SSIM_W * ssim(pred, yb)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
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

        # Validation: evaluate on a small subset of full volumes to save time
        model.eval()
        with torch.no_grad():
            vloss_acc = 0.0
            vcount = 0
            if val_patients:
                # Sample up to N patients per epoch for faster feedback
                N_VAL = min(4, len(val_patients))
                # rotate through patients across epochs
                start = ((epoch - 1) * N_VAL) % len(val_patients)
                sel = [val_patients[(start + i) % len(val_patients)] for i in range(N_VAL)]
                for vp in tqdm(sel, desc="Validation", unit="case", leave=False):
                    xt = torch.from_numpy(vp.input_arr).float()[None, None].to(DEVICE, non_blocking=True)
                    yt = torch.from_numpy(vp.target_arr).float()[None, None].to(DEVICE, non_blocking=True)
                    pred_full = inferer(xt, model)
                    lv = LOSS_MSE_W * mse(pred_full, yt) + LOSS_SSIM_W * ssim(pred_full, yt)
                    vloss_acc += float(lv.item())
                    vcount += 1
            val_loss = (vloss_acc / max(1, vcount)) if vcount > 0 else float("nan")
            val_hist.append(val_loss)

        dt = time.time() - t0
        skip_msg = f" | skipped_empty={skipped_empty}/{total_seen}" if total_seen > 0 else ""
        print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.4f} val={val_loss:.4f} ({dt:.1f}s){skip_msg}")
        try:
            plot_curves(train_hist, val_hist, CURVE_PNG)
        except Exception:
            pass

        # Save checkpoint every epoch
        try:
            ckpt_path = os.path.join(MODEL_DIR, f"dynunet_{RUN_ID}_epoch{epoch:03d}.pt")
            torch.save({
                'model_state': model.state_dict(),
                'run_id': RUN_ID,
                'epoch': epoch,
                'config': {
                    'patch_size': PATCH_SIZE,
                    'patch_stride': PATCH_STRIDE,
                    'epochs': EPOCHS,
                    'lr': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'loss_weights': (LOSS_MSE_W, LOSS_SSIM_W),
                }
            }, ckpt_path)
        except Exception:
            # best-effort checkpointing; continue training on failure
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
    ssim_eval = SSIMLoss(spatial_dims=3, data_range=1.0)
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
            out_path = os.path.join(OUTPUT_ROOT, f"{p.pid}_{RUN_ID}.nii.gz")
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
            # vmin = float(min(x_cor.min(), y_cor.min(), p_cor.min()))
            # vmax = float(max(x_cor.max(), y_cor.max(), p_cor.max()))
            vmin = 0. 
            vmax = 1.0 

            # Errors
            e_in = y_cor - x_cor
            e_out = y_cor - p_cor
            # eabs = float(max(abs(e_in).max(), abs(e_out).max(), 1e-6))
            eabs = 1.0

            # Build a compact 2x3 grid (bottom-right left blank)
            fig = plt.figure(figsize=(10, 6), dpi=120)
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.02, hspace=0.08)
            ax_in = fig.add_subplot(gs[0, 0])
            ax_tg = fig.add_subplot(gs[0, 1])
            ax_out = fig.add_subplot(gs[0, 2])
            ax_ein = fig.add_subplot(gs[1, 0])
            ax_eout = fig.add_subplot(gs[1, 1])
            ax_empty = fig.add_subplot(gs[1, 2])

            im_in = ax_in.imshow(x_cor.T, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
            ax_in.set_title('Input')
            im_tg = ax_tg.imshow(y_cor.T, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
            ax_tg.set_title('Target')
            im_out = ax_out.imshow(p_cor.T, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
            ax_out.set_title('Output')
            im_ein = ax_ein.imshow(e_in.T, cmap='bwr', vmin=-eabs, vmax=eabs, origin='lower')
            ax_ein.set_title('Err T-Input')
            im_eout = ax_eout.imshow(e_out.T, cmap='bwr', vmin=-eabs, vmax=eabs, origin='lower')
            ax_eout.set_title('Err T-Output')
            # Add colorbars for each image
            fig.colorbar(im_in, ax=ax_in, fraction=0.046, pad=0.04)
            fig.colorbar(im_tg, ax=ax_tg, fraction=0.046, pad=0.04)
            fig.colorbar(im_out, ax=ax_out, fraction=0.046, pad=0.04)
            fig.colorbar(im_ein, ax=ax_ein, fraction=0.046, pad=0.04)
            fig.colorbar(im_eout, ax=ax_eout, fraction=0.046, pad=0.04)
            for ax in (ax_in, ax_tg, ax_out, ax_ein, ax_eout, ax_empty):
                ax.axis('off')
            ax_empty.set_visible(False)
            fig.suptitle(p.pid)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_png = os.path.join(FIG_DIR, f"{RUN_ID}_{p.pid}_coronal.png")
            plt.savefig(out_png)
            plt.close(fig)
            print(f"Saved validation example plot: {out_png}")

            # Sagittal slice: fix Y (H axis) at middle
            ymid = xnp.shape[1] // 2
            x_sag = xnp[:, ymid, :]
            y_sag = ynp[:, ymid, :]
            p_sag = pnp[:, ymid, :]

            # vmin_s = float(min(x_sag.min(), y_sag.min(), p_sag.min()))
            # vmax_s = float(max(x_sag.max(), y_sag.max(), p_sag.max()))
            vmin_s = 0.
            vmax_s = 1.0

            e_in_s = y_sag - x_sag
            e_out_s = y_sag - p_sag
            # eabs_s = float(max(abs(e_in_s).max(), abs(e_out_s).max(), 1e-6))
            eabs_s = 1.0

            fig_s = plt.figure(figsize=(10, 6), dpi=120)
            gs_s = fig_s.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.02, hspace=0.08)
            ax_in_s = fig_s.add_subplot(gs_s[0, 0])
            ax_tg_s = fig_s.add_subplot(gs_s[0, 1])
            ax_out_s = fig_s.add_subplot(gs_s[0, 2])
            ax_ein_s = fig_s.add_subplot(gs_s[1, 0])
            ax_eout_s = fig_s.add_subplot(gs_s[1, 1])
            ax_empty_s = fig_s.add_subplot(gs_s[1, 2])

            im_in_s = ax_in_s.imshow(x_sag.T, cmap='inferno', vmin=vmin_s, vmax=vmax_s, origin='lower')
            ax_in_s.set_title('Input (Sagittal)')
            im_tg_s = ax_tg_s.imshow(y_sag.T, cmap='inferno', vmin=vmin_s, vmax=vmax_s, origin='lower')
            ax_tg_s.set_title('Target (Sagittal)')
            im_out_s = ax_out_s.imshow(p_sag.T, cmap='inferno', vmin=vmin_s, vmax=vmax_s, origin='lower')
            ax_out_s.set_title('Output (Sagittal)')
            im_ein_s = ax_ein_s.imshow(e_in_s.T, cmap='bwr', vmin=-eabs_s, vmax=eabs_s, origin='lower')
            ax_ein_s.set_title('Err T-Input')
            im_eout_s = ax_eout_s.imshow(e_out_s.T, cmap='bwr', vmin=-eabs_s, vmax=eabs_s, origin='lower')
            ax_eout_s.set_title('Err T-Output')
            fig_s.colorbar(im_in_s, ax=ax_in_s, fraction=0.046, pad=0.4/10)
            fig_s.colorbar(im_tg_s, ax=ax_tg_s, fraction=0.046, pad=0.4/10)
            fig_s.colorbar(im_out_s, ax=ax_out_s, fraction=0.046, pad=0.4/10)
            fig_s.colorbar(im_ein_s, ax=ax_ein_s, fraction=0.046, pad=0.4/10)
            fig_s.colorbar(im_eout_s, ax=ax_eout_s, fraction=0.046, pad=0.4/10)
            for ax in (ax_in_s, ax_tg_s, ax_out_s, ax_ein_s, ax_eout_s, ax_empty_s):
                ax.axis('off')
            ax_empty_s.set_visible(False)
            fig_s.suptitle(p.pid)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_png_s = os.path.join(FIG_DIR, f"{RUN_ID}_{p.pid}_sagittal.png")
            plt.savefig(out_png_s)
            plt.close(fig_s)
            print(f"Saved validation example plot: {out_png_s}")


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
    parser.add_argument("--model", type=str, choices=["DynUNet", "nnFormer"], default=MODEL_NAME, help="Model architecture to use")
    args = parser.parse_args()

    # Update run identifier to include DRF, and derived artifact paths
    global RUN_ID, CURVE_PNG
    RUN_ID = f"{RUN_ID_BASE}_drf{args.drf}"
    CURVE_PNG = os.path.join(FIG_DIR, f"training_curves_{RUN_ID}.png")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Ensure Zarr cache exists; convert if first run
    convert_all_to_zarr_if_needed(DATA_ROOT, ZARR_ROOT, META_CSV)

    # Build metadata index
    meta_idx = load_metadata_index(META_CSV)

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

    # Build Zarr entries and lazy patch dataset for training
    train_entries = build_zarr_entries(meta_idx, train_pids, args.drf, ZARR_ROOT)
    val_entries = build_zarr_entries(meta_idx, val_pids, args.drf, ZARR_ROOT)
    if not train_entries:
        print("No training patients available with required NPY pairs.")
        return

    train_ds = LazyPatchDataset(train_entries, PATCH_SIZE, PATCH_STRIDE, augment=True)
    print(f"Train patches: {len(train_ds)} across {len(train_entries)} patients")

    # Build full volumes for validation (smaller set) just-in-time
    print(f"Val patients: {len(val_entries)}")
    
    val_patients: List[PatientItem] = []
    for e in val_entries:
        try:
            val_patients.append(load_patient_from_zarr(ZARR_ROOT, meta_idx, e.pid, args.drf))
        except Exception as ex:
            print(f"Skipping val {e.pid}: {ex}")

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
