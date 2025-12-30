""" signal processing - filtering, resampling, normalization """
import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PreprocessConfig:
    """Configuration for basic sEMG preprocessing."""
    fs_in: float                 # original sampling rate (Hz)
    fs_target: float = 2000.0    # target sampling rate after resampling
    bandpass_low: float = 20.0   # band-pass low cutoff (Hz)
    bandpass_high: float = 450.0 # band-pass high cutoff (Hz)
    notch_freq: Optional[float] = 60.0  # 50 or 60 Hz, or None to disable
    notch_q: float = 30.0
    z_normalize: bool = True     # per-channel z-score


def bandpass_filter(x: np.ndarray,
                    fs: float,
                    low: float,
                    high: float,
                    order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to multichannel data.

    Args:
        x: array of shape [T, C] (time x channels)
        fs: sampling rate in Hz
        low, high: cutoff frequencies in Hz
        order: filter order

    Returns:
        Filtered array of shape [T, C].
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq

    sos = signal.butter(
        N=order,
        Wn=[low_norm, high_norm],
        btype="bandpass",
        output="sos",  # second-order sections (more numerically stable)
    )
    # axis=0 => filter along time dimension
    return signal.sosfiltfilt(sos, x, axis=0)

def notch_filter(x: np.ndarray,
                 fs: float,
                 freq: float = 60.0,
                 q: float = 30.0) -> np.ndarray:
    """
    Apply a notch filter around a given frequency (e.g. to remove 50/60 Hz mains hum).

    Args:
        x: array of shape [T, C]
        fs: sampling rate
        freq: notch center frequency in Hz
        q: quality factor (higher = narrower notch)

    Returns:
        Filtered array of shape [T, C].
    """
    nyq = fs / 2.0
    w0 = freq / nyq  # normalized frequency

    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, x, axis=0)

def resample_signal(x: np.ndarray,
                    fs_in: float,
                    fs_target: float) -> np.ndarray:
    """
    Resample multichannel data from fs_in to fs_target.
    Uses Fourier-based resampling.

    Args:
        x: array of shape [T, C]
        fs_in: original sampling rate
        fs_target: desired sampling rate

    Returns:
        Resampled array of shape [T_new, C].
    """
    if np.isclose(fs_in, fs_target):
        return x

    t, _ = x.shape
    t_new = int(round(t * fs_target / fs_in))
    return signal.resample(x, t_new, axis=0)


def zscore_per_channel(x: np.ndarray,
                       eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel z-normalization: (x - mean) / std, along time.

    Args:
        x: array of shape [T, C]
        eps: small constant to avoid division by zero

    Returns:
        x_norm: normalized data [T, C]
        mean: per-channel mean [C]
        std: per-channel std [C]
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0) + eps
    x_norm = (x - mean) / std
    return x_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)



