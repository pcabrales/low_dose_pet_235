# low_dose_pet_235

model training + inference for low-dose PET denoising.

Data layout assumed:
- Inputs at `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/all_subjects/<patient>/1-10 dose`
- Targets at `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/all_subjects/<patient>/Full_dose`
- Outputs written to `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/<patient>.nii.gz`

## Install

Requires Python 3.9+.

Recommended: create and activate a virtual environment, then install project deps from `pyproject.toml`:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Notes
- This installs `torch`, `monai`, `SimpleITK`, and other dependencies. Check `pyproject.toml` for details.
- If you need a specific CUDA build of PyTorch, follow the instructions at https://pytorch.org/get-started/locally/ before running `pip install -e .`.

## Run

Run the training + validation script directly. Specify the dose reduction factor (DRF) for the input folder name `1-<DRF> dose`:

```
python3 low_dose_pet_235/train.py --drf 10
```

Flags
- `--drf {2,4,10,20,50,100}`: selects the input folder `1-<DRF> dose` within each patient directory. Default: `10`.
- `--model {DynUNet,nnFormer}`: selects the model architecture. Default: `DynUNet`.
- `--validate`: runs validation only (no training).

What it does
- Trains model on 96x96x96 patches (stride 80) with flips/rotations.
- Loss: 0.9*MSE + 0.1*SSIM using AdamW.
- Saves training/validation curves to `low_dose_pet_235/figures/training_curves_<RUNID>.png`.
- Writes per-patient NIfTI outputs to `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output`.
- Prints baseline (input→target) vs model (output→target) MSE/SSIM and saves a CSV summary next to outputs.

Quick tips
- For a short test, edit `MAX_PATIENTS` and `EPOCHS` near the top of `low_dose_pet_235/train.py`.
- Ensure the expected data folders exist and contain a single DICOM/IMA series each.

## Artifacts and outputs

- Models: `low_dose_pet_235/trained-models/<MODEL_NAME>_<RUNID>.pt`
- Curves: `low_dose_pet_235/figures/training_curves_<RUNID>.png`
- Validation examples: `low_dose_pet_235/figures/<RUNID>_<PATIENT>_coronal.png`
- Metrics CSV: `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_<RUNID>.csv`

`<RUNID>` is the date and hour+minute of the run plus the DRF, formatted as `YYYYMMDD_HHMM_drf<DRF>`.
