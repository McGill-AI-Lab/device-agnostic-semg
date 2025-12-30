""" signal processing - filtering, resampling, normalization """
import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional

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
