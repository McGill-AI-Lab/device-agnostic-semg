import numpy as np
from typing import Tuple

def _compute_window_indices(
    n_samples: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Compute [start, end) indices for sliding windows over a 1D time axis.

    Returns:
        indices: array of shape [N, 2], each row is [start, end]
    """
    if window_size > n_samples:
        return np.zeros((0, 2), dtype=int)

    starts = np.arange(0, n_samples - window_size + 1, stride, dtype=int)
    ends = starts + window_size
    return np.stack([starts, ends], axis=1)


def make_emg_pose_windows(
    emg: np.ndarray,
    pose: np.ndarray,
    fs: float,
    window_ms: float,
    stride_ms: float,
    label_mode: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice continuous EMG + pose into overlapping windows.

    Args:
        emg: EMG array [T, C] at sampling rate fs
        pose: pose array [T, D] time-aligned with emg
        fs: sampling rate in Hz
        window_ms: window length in milliseconds
        stride_ms: stride (hop) between windows in milliseconds
        label_mode:
            - "center": take pose at center time of the window
            - "mean": average pose over the window

    Returns:
        X: EMG windows [N, T_win, C]
        y: pose labels [N, D]
    """
    assert emg.shape[0] == pose.shape[0], "EMG and pose must have same length."

    # Convert milliseconds to samples
    window_size = int(round(window_ms * fs / 1000.0))
    stride = int(round(stride_ms * fs / 1000.0))

    idx = _compute_window_indices(emg.shape[0], window_size, stride)
    if idx.shape[0] == 0:
        # No full window fits; return empty arrays
        return (
            np.zeros((0, window_size, emg.shape[1]), dtype=np.float32),
            np.zeros((0, pose.shape[1]), dtype=np.float32),
        )

    X_list = []
    y_list = []

    for start, end in idx:
        X_list.append(emg[start:end, :])

        if label_mode == "center":
            center = (start + end) // 2
            y_list.append(pose[center])
        elif label_mode == "mean":
            y_list.append(pose[start:end].mean(axis=0))
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N, T_win, C]
    y = np.stack(y_list, axis=0).astype(np.float32)  # [N, D]
    return X, y

