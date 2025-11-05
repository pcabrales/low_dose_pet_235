# Ultra-Low Dose PET Imaging Challenge at IEEE MIC 2025: team 235 (Universidad Complutense de Madrid)

Challenge webpage [link](https://udpet-challenge.github.io/).

Abstract [link](https://docs.google.com/document/d/1neA3LrznxqXWd78N11c6olObeVPOOW2nzeqWqZe_vmg/edit?usp=sharing).


## Training approaches

### Adaptive
The code for the noise-adaptive approach (nnFormer-FiLM) is found in folders `film_nnformer_uexplorer` and `film_nnformer_quadra`. You will need to install nnFormer from [here](https://github.com/282857341/nnFormer).

### Dedicated
The rest of the code is dedicated to the multi-model, noise-specific approach (nnFormer-composite). It includes the relevant nnFormer architecture within the `nnFormer` folder.


#### Install

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

#### Run

Run the training + validation script directly. Specify the dose reduction factor (DRF) for the input folder.

For the Siemens Quadra data:
```
python3 low_dose_pet_235/train.py --drf 10
```

For the United Explorer data:

```
python3 low_dose_pet_235/train_explorer.py --drf 10 
```

Flags
- `--drf {2,4,10,20,50,100}`: selects the input folder `1-<DRF> dose` within each patient directory. Default: `10`.
- `--model {DynUNet,nnFormer}`: selects the model architecture. Default: `DynUNet`.
- `--validate`: runs validation only (no training).

What it does
- Trains model on 96x96x96 patches (stride 80) with flips/rotations.
- Loss: 0.95*MSE + 0.05*SSIM using AdamW.
- Saves training/validation curves to `low_dose_pet_235/figures/training_curves_<RUNID>.png`.
- Writes per-patient NIfTI outputs to `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output`.
- Prints baseline (input→target) vs model (output→target) MSE/SSIM and saves a CSV summary next to outputs.

Quick tips
- For a short test, edit `MAX_PATIENTS` and `EPOCHS` near the top of `low_dose_pet_235/train.py`.
- Ensure the expected data folders exist and contain a single DICOM/IMA series each.

#### Outputs

- Models: `low_dose_pet_235/trained-models/<MODEL_NAME>_<RUNID>.pt`
- Curves: `low_dose_pet_235/figures/training_curves_<MODEL_NAME>_<RUNID>.png`
- Validation examples: `low_dose_pet_235/figures/<MODEL_NAME>_<RUNID>_<PATIENT>_coronal.png`
- Metrics CSV: `/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_<MODEL_NAME>_<RUNID>.csv`

`<RUNID>` is the date and hour+minute of the run plus the DRF, formatted as `YYYYMMDD_HHMM_drf<DRF>`.

#### Inference
Run the test script specifying the DRF. Define the pretrained model in `low_dose_pet_235/train.py`.

For the Siemens Quadra data:
```
python3 low_dose_pet_235/test.py --drf 10
```
For the United Explorer data:

```
python3 low_dose_pet_235/test_explorer.py --drf 10
```
