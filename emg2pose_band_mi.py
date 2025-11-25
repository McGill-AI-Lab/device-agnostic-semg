import os
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_regression

# ============================================================
# CONFIG
# ============================================================

# Root directory where your *.hdf5 files live
DATA_ROOT = Path("/home/klambert/projects/aip-craffel/klambert/sEMG/emg2pose_dataset_mini")  # <-- change this

# EMG & pose come from a structured dataset:
# group "emg2pose" → dataset "timeseries" → fields "emg", "joint_angles"
GROUP_NAME = "emg2pose"
TIMESERIES_NAME = "timeseries"
EMG_FIELD = "emg"
POSE_FIELD = "joint_angles"

# Sampling frequency (Hz)
FS = 2000

# Frequency band of interest
FMIN = 500.0
FMAX = 850.0

# Windowing (samples)
WINDOW_SIZE = 256      # ~128 ms
STEP_SIZE = 128        # 50% overlap

# Limits (to keep it fast)
MAX_FILES = 5
MAX_WINDOWS_TOTAL = 20000

# Which pose dimension to use as the label
POSE_DIM = 0  # 0..19 since joint_angles has shape (T, 20)


# ============================================================
# Load EMG + pose from one file
# ============================================================

def load_emg_and_pose(h5_path: Path):
    """
    Load EMG (T, 16) and pose (T, 20) from
    group 'emg2pose' / dataset 'timeseries' structured array.
    """
    with h5py.File(h5_path, "r") as f:
        ts = f[GROUP_NAME][TIMESERIES_NAME][...]   # shape (T,), structured dtype
        # Each element has fields: 'time', 'joint_angles', 'emg'
        emg = np.array(ts[EMG_FIELD], dtype=np.float32)           # (T, 16)
        pose = np.array(ts[POSE_FIELD], dtype=np.float32)         # (T, 20)
    return emg, pose


# ============================================================
# Bandpower features 500–850 Hz
# ============================================================

def extract_bandpower_features(emg: np.ndarray,
                               fs: int = FS,
                               fmin: float = FMIN,
                               fmax: float = FMAX,
                               window_size: int = WINDOW_SIZE,
                               step_size: int = STEP_SIZE):
    """
    emg: (T, C) array
    returns:
        X   : (num_windows, C) bandpower in [fmin, fmax]
        idx : (num_windows,) center indices for label alignment
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


# ============================================================
# Label extraction
# ============================================================

def extract_labels_for_windows(pose: np.ndarray,
                               centers: np.ndarray,
                               pose_dim: int = POSE_DIM):
    """
    pose: (T, D)
    centers: (num_windows,) indices
    returns:
        y: (num_windows,) pose scalar
    """
    T, D = pose.shape
    assert 0 <= pose_dim < D
    centers = np.clip(centers, 0, T - 1)
    y = pose[centers, pose_dim]
    return y.astype(np.float32)


# ============================================================
# Collect dataset from multiple files
# ============================================================

def collect_dataset(root: Path,
                    max_files: int = MAX_FILES,
                    max_windows_total: int = MAX_WINDOWS_TOTAL):
    """
    Scan for .hdf5 files, extract bandpower features + labels.
    """
    files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files under {root}")

    print(f"Found {len(files)} HDF5 files")
    files = files[:max_files]
    print(f"Using first {len(files)} files")

    X_list = []
    y_list = []
    total_windows = 0

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path}")
        emg, pose = load_emg_and_pose(path)
        print(f"  emg shape:  {emg.shape}")
        print(f"  pose shape: {pose.shape}")

        X_file, centers = extract_bandpower_features(emg)
        y_file = extract_labels_for_windows(pose, centers)

        remaining = max_windows_total - total_windows
        if remaining <= 0:
            break

        if X_file.shape[0] > remaining:
            X_file = X_file[:remaining]
            y_file = y_file[:remaining]

        X_list.append(X_file)
        y_list.append(y_file)

        total_windows += X_file.shape[0]
        print(f"  collected {X_file.shape[0]} windows (total {total_windows})")

        if total_windows >= max_windows_total:
            break

    if not X_list:
        raise RuntimeError("No windows collected – check paths/settings")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"\nFinal dataset: X={X_all.shape}, y={y_all.shape}")
    return X_all, y_all


# ============================================================
# Mutual information
# ============================================================

def compute_mutual_information(X: np.ndarray, y: np.ndarray):
    """
    MI between each channel's bandpower and pose scalar.
    """
    # log transform to stabilize then z-score
    X_log = np.log1p(X)
    X_mean = X_log.mean(axis=0, keepdims=True)
    X_std = X_log.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_log - X_mean) / X_std

    print("\nComputing mutual information...")
    mi = mutual_info_regression(X_norm, y, random_state=0)
    return mi


# ============================================================
# Main
# ============================================================

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"{DATA_ROOT} does not exist – update DATA_ROOT")

    X, y = collect_dataset(DATA_ROOT)
    mi = compute_mutual_information(X, y)

    print("\n=== Mutual information per EMG channel (500–850 Hz bandpower vs pose_dim) ===")
    for ch, val in enumerate(mi):
        print(f"Channel {ch:2d}: MI ≈ {val:.4f}")

    print(f"\nTotal MI (sum over channels): {mi.sum():.4f}")
    print("You can change POSE_DIM or FMIN/FMAX at the top and re-run to compare bands/labels.")


if __name__ == "__main__":
    main()
