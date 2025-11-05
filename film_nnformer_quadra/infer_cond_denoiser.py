#!/usr/bin/env python3
"""
Run nnFormer denoising on a NIfTI volume.

Example:
    python infer_cond_denoiser.py \
        --input_nii noisy.nii.gz \
        --output_nii denoised.nii.gz \
        --noise_level 0.1
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference


def import_nnformer():
    """Import the conditioned nnFormer implementation, trying common locations."""
    tried = []
    try:
        from nnformer.network_architecture.nnFormer_synapse import nnFormer as NNFormer
        return NNFormer
    except Exception as exc:  # pragma: no cover - informative message on failure
        tried.append(("nnformer.network_architecture.nnFormer_synapse", str(exc)))
    try:
        from nnformer.network_architecture.nnFormer import nnFormer as NNFormer
        return NNFormer
    except Exception as exc:
        tried.append(("nnformer.network_architecture.nnFormer", str(exc)))
    msg = "Could not import nnFormer. Tried:\n" + "\n".join([f"  {m}: {err}" for m, err in tried])
    raise ImportError(msg)


class CondNNFormerDenoiser(torch.nn.Module):
    def __init__(self, crop_size: Tuple[int, int, int], d_cond: int = 1):
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


def load_nifti(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, img.affine


def save_nifti(path: pathlib.Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def main():
    parser = argparse.ArgumentParser(description="Run conditioned nnFormer denoising on a NIfTI image.")
    parser.add_argument("--input_nii", required=True, help="Path to the noisy input NIfTI (.nii or .nii.gz)")
    parser.add_argument("--output_nii", required=True, help="Path to save the denoised NIfTI")
    parser.add_argument("--sigma", type=float, required=True, help="Noise sigma value used during training (e.g. 0.01, 0.1, 0.25, 1.0)")
    parser.add_argument("--weights", default="/RAID/mpet/PET_LOWDOSE/nnFormer/nnformer/cond_nnformer_denoiser_best.pt", help="Path to the trained weights (.pt state dict)")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--crop", default="96,96,96", help="Sliding-window patch size D,H,W")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap (0-1)")
    parser.add_argument("--sw_batch_size", type=int, default=1, help="Sliding window batch size")
    parser.add_argument("--no_intensity_norm", action="store_true", help="Disable global min-max normalisation")
    parser.add_argument("--disable_amp", action="store_true", help="Disable autocast mixed precision on CUDA")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_nii)
    output_path = pathlib.Path(args.output_nii)
    if not input_path.exists():
        raise SystemExit(f"Input NIfTI not found: {input_path}")

    print(f"Loading {input_path} ...", flush=True)
    volume, affine = load_nifti(input_path)

    vmin = float(volume.min())
    vmax = float(volume.max())
    denom = vmax - vmin
    if args.no_intensity_norm or denom <= 0:
        volume_norm = volume.copy()
        inverse_scale = None
    else:
        volume_norm = (volume - vmin) / denom
        inverse_scale = (vmin, denom)

    crop = tuple(int(v) for v in args.crop.split(","))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    weights_path = pathlib.Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    print(f"Loading model weights from {weights_path}", flush=True)
    model = CondNNFormerDenoiser(crop_size=crop, d_cond=1).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    sigma_tensor = torch.tensor([[args.sigma]], dtype=torch.float32, device=device)

    input_tensor = torch.from_numpy(volume_norm[None, None]).to(device)

    def predictor(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cond = sigma_tensor.expand(b, -1)
        return model(x, cond)

    print("Running sliding-window inference ...", flush=True)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        pred = sliding_window_inference(
            inputs=input_tensor,
            roi_size=crop,
            sw_batch_size=args.sw_batch_size,
            predictor=predictor,
            overlap=args.overlap,
            mode="gaussian",
        )

    output = pred.float().cpu().numpy()[0, 0]
    if inverse_scale is not None:
        vmin, denom = inverse_scale
        output = output * denom + vmin

    print(f"Saving denoised volume to {output_path}", flush=True)
    save_nifti(output_path, output, affine)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
