import os
import csv
from typing import List, Tuple

import numpy as np


def load_column_stats(csv_path: str, columns: List[str]) -> Tuple[str, dict]:
    """Load specified columns from a CSV and compute mean/std.

    Returns a tuple of (path, stats) where stats is {col: (mean, std)}.
    """
    stats = {}
    values = {c: [] for c in columns}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"Missing columns in {csv_path}: {', '.join(missing)}; "
                f"available: {', '.join(reader.fieldnames or [])}"
            )
        for row in reader:
            for c in columns:
                v = row.get(c, "")
                if v is None or v == "":
                    continue
                try:
                    values[c].append(float(v))
                except ValueError:
                    # Skip non-numeric entries gracefully
                    continue

    for c in columns:
        arr = np.asarray(values[c], dtype=float)
        if arr.size == 0:
            stats[c] = (float("nan"), float("nan"))
        else:
            stats[c] = (float(arr.mean()), float(arr.std(ddof=0)))

    return csv_path, stats


def print_stats_for_files(paths: List[str], columns: List[str]) -> None:
    for p in paths:
        csv_path = os.path.expanduser(p)
        path_label = csv_path
        try:
            csv_path, stats = load_column_stats(csv_path, columns)
        except Exception as e:
            print(f"[ERROR] {path_label}: {e}")
            continue

        print(f"\nFile: {path_label}")
        for c in columns:
            mean, std = stats[c]
            print(f"  {c}: mean={mean:.6f}  std={std:.6f}")


if __name__ == "__main__":
    # files = [
    #     "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_20251103_1019_drf10.csv",
    #     "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_20251103_1109_drf10.csv",
    #     "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_20251103_1115_drf10.csv",
    # ]
    files = [
        "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_20251103_1041_drf50.csv",
        "/root/PET_LOWDOSE/TRAINING_DATA/Bern-Inselspital-2022/output/metrics_summary_dynunet_20251104_0830_drf50.csv",
    ]
    cols = ["model_ssim", "model_mse"]
    print_stats_for_files(files, cols)
