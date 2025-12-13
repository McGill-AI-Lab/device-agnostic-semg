import os
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_regression

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("/home/klambert/projects/aip-craffel/klambert/sEMG/emg2pose_dataset_mini") 

GROUP_NAME = "emg2pose"
TIMESERIES_NAME = "timeseries"
EMG_FIELD = "emg"
POSE_FIELD = "joint_angles"

FS = 2000                   # sampling frequency (Hz)
WINDOW_SIZE = 256           # samples (~128 ms)
STEP_SIZE = 128             # 50% overlap

MAX_FILES = 5
MAX_WINDOWS_TOTAL = 20000

POSE_DIM = 0                # which joint angle to use as label

# 100 Hz bins from 0–1000 Hz
BANDS_100 = [
    (f"{lo:03d}-{hi:03d}", float(lo), float(hi))
    for lo, hi in zip(range(0, 1000, 100), range(100, 1100, 100))
]   # ("000-100", 0, 100), ("100-200", 100, 200), ... , ("900-1000", 900, 1000)

# Special band 500–850 Hz
SPECIAL_BANDS = [
    ("500-850", 500.0, 850.0),
]


# ============================================================
# DATA LOADER
# ============================================================

def load_emg_and_pose(h5_path: Path):
    """
    Load EMG (T, 16) and pose (T, 20) from
    group 'emg2pose' / dataset 'timeseries' structured array.
    """
    with h5py.File(h5_path, "r") as f:
        ts = f[GROUP_NAME][TIMESERIES_NAME][...]   # shape (T,), structured dtype
        emg = np.array(ts[EMG_FIELD], dtype=np.float32)         # (T, 16)
        pose = np.array(ts[POSE_FIELD], dtype=np.float32)       # (T, 20)
    return emg, pose


# ============================================================
# FEATURE / LABEL EXTRACTION
# ============================================================

def extract_bandpower_features(emg: np.ndarray,
                               fmin: float,
                               fmax: float,
                               fs: int = FS,
                               window_size: int = WINDOW_SIZE,
                               step_size: int = STEP_SIZE):
    """
    For each window and channel, compute bandpower in [fmin, fmax] Hz.
    Returns:
        X   : (num_windows, C)
        idx : (num_windows,) center indices (to align labels)
    """
    T, C = emg.shape
    feats = []
    centers = []

    start = 0
    while start + window_size <= T:
        end = start + window_size
        segment = emg[start:end, :]   # (window_size, C)

        bandpowers = []
        for ch in range(C):
            freqs, psd = welch(
                segment[:, ch],
                fs=fs,
                nperseg=window_size,
                noverlap=window_size // 2,
                detrend="constant",
                scaling="density",
            )
            mask = (freqs >= fmin) & (freqs <= fmax)
            bandpowers.append(psd[mask].sum())

        feats.append(bandpowers)
        centers.append(start + window_size // 2)
        start += step_size

    X = np.asarray(feats, dtype=np.float32)
    idx = np.asarray(centers, dtype=np.int64)
    return X, idx


def extract_labels_for_windows(pose: np.ndarray,
                               centers: np.ndarray,
                               pose_dim: int = POSE_DIM):
    T, D = pose.shape
    assert 0 <= pose_dim < D
    centers = np.clip(centers, 0, T - 1)
    y = pose[centers, pose_dim]
    return y.astype(np.float32)


# ============================================================
# DATASET BUILDING FOR A GIVEN BAND
# ============================================================

def collect_dataset_for_band(root: Path,
                             fmin: float,
                             fmax: float,
                             max_files: int = MAX_FILES,
                             max_windows_total: int = MAX_WINDOWS_TOTAL):
    """
    For a given frequency band [fmin, fmax], build (X, y) over several files.
    """
    files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under {root}")

    files = files[:max_files]
    print(f"  Using {len(files)} files for band {fmin}-{fmax} Hz")

    X_list = []
    y_list = []
    total_windows = 0

    for path in files:
        emg, pose = load_emg_and_pose(path)

        X_file, centers = extract_bandpower_features(emg, fmin=fmin, fmax=fmax)
        y_file = extract_labels_for_windows(pose, centers, pose_dim=POSE_DIM)

        remaining = max_windows_total - total_windows
        if remaining <= 0:
            break

        if X_file.shape[0] > remaining:
            X_file = X_file[:remaining]
            y_file = y_file[:remaining]

        X_list.append(X_file)
        y_list.append(y_file)

        total_windows += X_file.shape[0]

        if total_windows >= max_windows_total:
            break

    if not X_list:
        raise RuntimeError(f"No windows collected for band {fmin}-{fmax}")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    return X_all, y_all


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def compute_mutual_information(X: np.ndarray, y: np.ndarray):
    """
    MI between each channel's bandpower and pose scalar.
    """
    # stabilize scale: log1p + z-score
    X_log = np.log1p(X)
    X_mean = X_log.mean(axis=0, keepdims=True)
    X_std = X_log.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_log - X_mean) / X_std

    mi = mutual_info_regression(X_norm, y, random_state=0)
    return mi


# ============================================================
# MAIN: SWEEP BANDS
# ============================================================

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"{DATA_ROOT} does not exist – update DATA_ROOT at the top")

    # All bands we want to test:
    all_bands = BANDS_100 + SPECIAL_BANDS

    results = []  # (name, fmin, fmax, total_mi, mean_mi, per_channel_mi)

    for name, fmin, fmax in all_bands:
        print(f"\n=== Band {name} Hz ({fmin}-{fmax}) ===")
        X, y = collect_dataset_for_band(DATA_ROOT, fmin, fmax)
        mi = compute_mutual_information(X, y)

        total_mi = float(mi.sum())
        mean_mi = float(mi.mean())

        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  per-channel MI: {', '.join(f'{v:.3f}' for v in mi)}")
        print(f"  TOTAL MI across channels: {total_mi:.4f}")
        print(f"  MEAN MI across channels:  {mean_mi:.4f}")

        results.append((name, fmin, fmax, total_mi, mean_mi, mi))

    # After running, you can interpret:
    #   - which 100 Hz bins carry most MI
    #   - how the 500–850 band compares to the sum/mean of others

    print("\n=== Summary (by band) ===")
    for name, fmin, fmax, total_mi, mean_mi, mi in results:
        print(f"{name:8s} [{fmin:5.1f}-{fmax:5.1f}] Hz  -> total MI={total_mi:.4f}, mean MI={mean_mi:.4f}")


if __name__ == "__main__":
    main()
