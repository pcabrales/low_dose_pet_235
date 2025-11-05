from __future__ import annotations

#!/usr/bin/env python3
"""Denoise a volume from the FiLM-conditioned nnFormer."""

import argparse
import numpy as np
import torch
import zarr

try:
    import nibabel as nib  # optional, only if --save_nii is used
except Exception:  # pragma: no cover - optional dependency
    nib = None

from torch import nn

# Default dataset location for the provided environment
DEFAULT_ZARR = "/RAID/mpet/PET_LOWDOSE/TRAINING_DATA/quadra_pet.zarr"


def parse_sigma_from_key(key: str) -> float:
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


def import_nnformer():
    """Import the conditioned nnFormer implementation, trying common locations."""
    tried = []
    try:
        from nnformer.network_architecture.nnFormer_synapse import nnFormer as NNFormer
        return NNFormer
    except Exception as e:  # pragma: no cover - informative message on failure
        tried.append(("nnformer.network_architecture.nnFormer_synapse", str(e)))
    try:
        from nnformer.network_architecture.nnFormer import nnFormer as NNFormer
        return NNFormer
    except Exception as e:
        tried.append(("nnformer.network_architecture.nnFormer", str(e)))
    msg = "Could not import nnFormer. Tried:\n" + "\n".join([f"  {m}: {err}" for m, err in tried])
    raise ImportError(msg)


class CondNNFormerDenoiser(nn.Module):
    def __init__(self, crop_size: tuple[int, int, int], d_cond: int = 1):
        super().__init__()
        NNFormer = import_nnformer()
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


def collect_sigma_range(root: zarr.hierarchy.Group) -> tuple[float, float]:
    sigmas = []
    try:
        iter_keys = root.group_keys()
    except AttributeError:
        iter_keys = root.keys()
    for key in iter_keys:
        g = root[key]
        for name in g.keys():
            if str(name).startswith("noisy_"):
                try:
                    sigmas.append(parse_sigma_from_key(name))
                except Exception:
                    continue
    if not sigmas:
        raise RuntimeError("No noisy_* datasets found to infer sigma range.")
    return float(np.min(sigmas)), float(np.max(sigmas))


def main():
    parser = argparse.ArgumentParser(description="Run conditioned nnFormer denoising on a Zarr case.")
    parser.add_argument("--case", required=True, help="Case/group identifier inside the Zarr store")
    parser.add_argument("--level", required=True, help="Noise level suffix, e.g. 0.200 for noisy_0.200")
    parser.add_argument("--weights", default="cond_nnformer_denoiser_best.pt", help="Model weights path")
    parser.add_argument("--zarr", default=DEFAULT_ZARR, help="Path to the quadra_pet Zarr store")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--crop", default="96,96,96", help="Crop size used during training (D,H,W)")
    parser.add_argument("--save_nii", default="", help="Optional NIfTI output path")
    args = parser.parse_args()

    root_r = zarr.open(args.zarr, mode="r")
    available_cases = []
    try:
        available_cases = list(root_r.group_keys())
    except AttributeError:
        available_cases = list(root_r.keys())
    if args.case not in available_cases:
        raise SystemExit(f"Case '{args.case}' not found. Available: {available_cases}")

    group = root_r[args.case]
    noisy_key = f"noisy_{args.level}"
    if noisy_key not in group:
        raise SystemExit(f"Dataset '{noisy_key}' not found in case '{args.case}'. Available: {list(group.keys())}")

    clean = group.get("clean", None)
    noisy_ds = group[noisy_key]
    arr = np.asarray(noisy_ds[:], dtype=np.float32)
    arr = np.expand_dims(np.expand_dims(arr, 0), 0)  # [1,1,D,H,W]

    # Normalise sigma to [0,1] using dataset-wide min/max
    sigma_min, sigma_max = collect_sigma_range(root_r)
    sigma_val = parse_sigma_from_key(f"noisy_{args.level}")
    sigma_tensor = torch.tensor([[sigma_val]], dtype=torch.float32)

    crop = tuple(int(v) for v in args.crop.split(","))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CondNNFormerDenoiser(crop_size=crop, d_cond=sigma_tensor.shape[-1]).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    x = torch.from_numpy(arr).to(device)
    c = sigma_tensor.to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        pred = model(x, c)
    output = pred.float().cpu().numpy()[0, 0]

    # Save back to Zarr under denoised_<level>
    root_w = zarr.open(args.zarr, mode="a")
    group_w = root_w[args.case]
    save_key = f"denoised_{args.level}"
    if save_key in group_w:
        del group_w[save_key]
    compressor = noisy_ds.compressor
    chunking = noisy_ds.chunks
    group_w.create_dataset(save_key, data=output, chunks=chunking, compressor=compressor)
    print(f"Saved denoised volume to {args.case}/{save_key}")

    if args.save_nii:
        if nib is None:
            print("nibabel not available; skipping NIfTI export")
        else:
            affine = np.eye(4, dtype=np.float64)
            if clean is not None:
                aff_attr = clean.attrs.get("affine")
                if aff_attr is not None:
                    affine = np.array(aff_attr, dtype=np.float64)
            nib.save(nib.Nifti1Image(output, affine), args.save_nii)
            print(f"Saved NIfTI to {args.save_nii}")


if __name__ == "__main__":
    main()
