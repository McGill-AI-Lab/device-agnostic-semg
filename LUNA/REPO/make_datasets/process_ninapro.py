"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PROCESS_NINAPRO.PY - NINAPRO RAW DATA PREPROCESSING               ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Converts raw Ninapro .mat files into preprocessed HDF5 format suitable for 
   training LUNA-style models. Handles windowing, labeling, and optional 
   glove data extraction for pose regression.

SUPPORTED DATABASES:
   - DB1: 27 subjects, 10 channels, 52 gestures, 100 Hz
   - DB2: 40 subjects, 12 channels, 49 gestures, 2000 Hz
   - DB4: 10 subjects, 12 channels, 52 gestures, 2000 Hz
   - DB5: 10 subjects, 16 channels, 52 gestures, 200 Hz
   - DB7: 20 subjects, 12 channels, 40 gestures, 2000 Hz
   - DB8: 10 subjects, 16 channels, 9 gestures, 2000 Hz

PREPROCESSING STEPS:
   1. Load .mat file for each subject/exercise
   2. Extract EMG channels and optionally glove data
   3. Segment into fixed-size windows
   4. Label each window by majority gesture
   5. Filter and resample if needed
   6. Save to HDF5 format

USAGE:
   # Process Ninapro DB2 for classification
   python process_ninapro.py --db db2 --root_dir /data/ninapro/db2 \
       --output_dir /processed/ninapro --mode classify
   
   # Process DB8 for pose regression
   python process_ninapro.py --db db8 --root_dir /data/ninapro/db8 \
       --output_dir /processed/ninapro --mode regress

OUTPUT:
   /processed/ninapro/db2/
   ├── train.h5
   ├── val.h5
   └── test.h5
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Pool
import warnings

try:
    import scipy.io as sio
    from scipy.signal import butter, filtfilt, resample
except ImportError:
    print("scipy required: pip install scipy")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("h5py required: pip install h5py")
    sys.exit(1)

from tqdm import tqdm


# ============================================================================
# Ninapro Database Configurations
# ============================================================================

DB_CONFIGS = {
    'db1': {
        'num_channels': 10,
        'num_subjects': 27,
        'num_exercises': 3,
        'sampling_rate': 100,
        'has_glove': True,
        'glove_channels': 22,
        'file_pattern': 'S{subject}_E{exercise}_A1.mat',
    },
    'db2': {
        'num_channels': 12,
        'num_subjects': 40,
        'num_exercises': 3,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_channels': 22,
        'file_pattern': 'S{subject}_E{exercise}_A1.mat',
    },
    'db4': {
        'num_channels': 12,
        'num_subjects': 10,
        'num_exercises': 3,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_channels': 22,
        'file_pattern': 's{subject}/S{subject}_E{exercise}_A1.mat',
    },
    'db5': {
        'num_channels': 16,
        'num_subjects': 10,
        'num_exercises': 3,
        'sampling_rate': 200,
        'has_glove': True,
        'glove_channels': 22,
        'file_pattern': 's{subject}/S{subject}_E{exercise}_A1.mat',
    },
    'db7': {
        'num_channels': 12,
        'num_subjects': 22,
        'num_exercises': 3,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_channels': 18,
        'file_pattern': 's{subject}/S{subject}_E{exercise}_A1.mat',
    },
    'db8': {
        'num_channels': 16,
        'num_subjects': 12,
        'num_exercises': 1,  # Single exercise for finger movements
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_channels': 18,
        'file_pattern': 's{subject}/S{subject}_E{exercise}_A1.mat',
    },
}


# ============================================================================
# Signal Processing Functions
# ============================================================================

def bandpass_filter(data: np.ndarray, fs: float, lowcut: float = 20.0, 
                    highcut: float = 500.0, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to EMG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)  # Ensure valid range
    
    if low >= high:
        return data  # Skip filtering if invalid range
    
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter along time axis (axis 0 if data is [T, C])
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        return np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, data)
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


def normalize_emg(data: np.ndarray) -> np.ndarray:
    """Z-score normalize EMG per channel."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    return (data - mean) / std


def resample_signal(data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample signal to target sampling rate."""
    if abs(orig_fs - target_fs) < 1.0:
        return data
    
    num_samples = int(len(data) * target_fs / orig_fs)
    return resample(data, num_samples, axis=0)


# ============================================================================
# Windowing Functions
# ============================================================================

def extract_windows(
    emg: np.ndarray,
    labels: np.ndarray,
    glove: Optional[np.ndarray],
    window_size: int,
    stride: int,
    min_label_ratio: float = 0.5,
    include_rest: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[Optional[np.ndarray]]]:
    """
    Extract fixed-size windows from continuous signal.
    
    Args:
        emg: [T, C] EMG signal
        labels: [T] gesture labels
        glove: [T, D] glove data (optional)
        window_size: Window length in samples
        stride: Step between windows
        min_label_ratio: Minimum fraction of window that must have same label
        include_rest: Include windows labeled as rest (0)
    
    Returns:
        windows: List of [C, T] EMG windows
        window_labels: List of majority labels
        window_glove: List of [D] mean glove values (or None)
    """
    T = emg.shape[0]
    windows = []
    window_labels = []
    window_glove = []
    
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        
        # Get window data
        emg_window = emg[start:end]  # [window_size, C]
        label_window = labels[start:end]
        
        # Find majority label
        unique, counts = np.unique(label_window, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_label = unique[majority_idx]
        majority_ratio = counts[majority_idx] / window_size
        
        # Skip if label is not consistent enough
        if majority_ratio < min_label_ratio:
            continue
        
        # Skip rest if not including
        if not include_rest and majority_label == 0:
            continue
        
        # Transpose to [C, T] format
        windows.append(emg_window.T)
        window_labels.append(int(majority_label))
        
        # Get glove data if available
        if glove is not None:
            glove_window = glove[start:end]
            window_glove.append(glove_window.mean(axis=0))  # Mean across window
        else:
            window_glove.append(None)
    
    return windows, window_labels, window_glove


# ============================================================================
# Main Processing Functions
# ============================================================================

def load_mat_file(filepath: str, db_version: str) -> Dict[str, np.ndarray]:
    """Load a Ninapro .mat file and extract relevant fields."""
    try:
        mat = sio.loadmat(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}
    
    result = {}
    
    # EMG data
    if 'emg' in mat:
        result['emg'] = np.array(mat['emg'], dtype=np.float32)
    
    # Labels (stimulus or restimulus)
    if 'restimulus' in mat:
        result['labels'] = np.array(mat['restimulus']).squeeze()
    elif 'stimulus' in mat:
        result['labels'] = np.array(mat['stimulus']).squeeze()
    
    # Glove data
    if 'glove' in mat:
        result['glove'] = np.array(mat['glove'], dtype=np.float32)
    
    # Subject info
    if 'subject' in mat:
        result['subject'] = int(np.array(mat['subject']).squeeze())
    
    return result


def process_subject(
    subject_id: int,
    db_version: str,
    root_dir: str,
    output_dir: str,
    window_size_ms: float = 200.0,
    stride_ms: float = 50.0,
    target_fs: Optional[float] = None,
    apply_filter: bool = True,
    mode: str = 'classify',
    include_rest: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process all exercises for one subject.
    
    Args:
        subject_id: Subject number (1-indexed)
        db_version: Database version ('db1', 'db2', etc.)
        root_dir: Root directory containing raw .mat files
        output_dir: Output directory for processed files
        window_size_ms: Window size in milliseconds
        stride_ms: Stride in milliseconds
        target_fs: Target sampling rate (None = keep original)
        apply_filter: Apply bandpass filter
        mode: 'classify' or 'regress'
        include_rest: Include rest periods
    
    Returns:
        List of processed samples as dicts
    """
    config = DB_CONFIGS[db_version]
    samples = []
    
    for exercise in range(1, config['num_exercises'] + 1):
        # Build file path
        pattern = config['file_pattern']
        rel_path = pattern.format(subject=subject_id, exercise=exercise)
        filepath = os.path.join(root_dir, rel_path)
        
        if not os.path.exists(filepath):
            # Try lowercase 's' prefix
            alt_pattern = pattern.replace('S{subject}', 's{subject}')
            rel_path = alt_pattern.format(subject=subject_id, exercise=exercise)
            filepath = os.path.join(root_dir, rel_path)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        # Load data
        data = load_mat_file(filepath, db_version)
        if not data or 'emg' not in data or 'labels' not in data:
            print(f"Invalid data in {filepath}")
            continue
        
        emg = data['emg']  # [T, C]
        labels = data['labels']  # [T]
        glove = data.get('glove', None)  # [T, D] or None
        
        fs = config['sampling_rate']
        
        # Apply bandpass filter
        if apply_filter:
            emg = bandpass_filter(emg, fs, lowcut=20.0, highcut=min(450.0, fs/2 - 1))
        
        # Resample if needed
        if target_fs is not None and target_fs != fs:
            emg = resample_signal(emg, fs, target_fs)
            labels = resample_signal(labels.astype(float), fs, target_fs).astype(int)
            if glove is not None:
                glove = resample_signal(glove, fs, target_fs)
            fs = target_fs
        
        # Calculate window parameters in samples
        window_size = int(window_size_ms * fs / 1000)
        stride = int(stride_ms * fs / 1000)
        
        # Extract windows
        windows, window_labels, window_glove = extract_windows(
            emg, labels, glove if mode == 'regress' else None,
            window_size, stride,
            min_label_ratio=0.5,
            include_rest=include_rest,
        )
        
        # Create samples
        for i, (win, label, glv) in enumerate(zip(windows, window_labels, window_glove)):
            sample = {
                'X': win,  # [C, T]
                'y': label,
                'subject_id': subject_id,
                'exercise': exercise,
            }
            
            if mode == 'regress' and glv is not None:
                sample['pose'] = glv  # [D]
            
            samples.append(sample)
    
    return samples


def save_to_hdf5(samples: List[Dict], output_file: str, group_size: int = 1000):
    """Save processed samples to HDF5 file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with h5py.File(output_file, 'w') as h5f:
        for i in range(0, len(samples), group_size):
            batch = samples[i:i + group_size]
            grp = h5f.create_group(f"data_group_{i // group_size}")
            
            # Stack arrays
            X = np.stack([s['X'] for s in batch])
            y = np.array([s['y'] for s in batch])
            subject_ids = np.array([s['subject_id'] for s in batch])
            
            grp.create_dataset('X', data=X, dtype='float32')
            grp.create_dataset('y', data=y, dtype='int64')
            grp.create_dataset('subject_id', data=subject_ids, dtype='int64')
            
            # Optional pose data
            if 'pose' in batch[0]:
                pose = np.stack([s['pose'] for s in batch])
                grp.create_dataset('pose', data=pose, dtype='float32')
    
    print(f"Saved {len(samples)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process Ninapro dataset")
    parser.add_argument('--db', type=str, required=True, 
                        choices=list(DB_CONFIGS.keys()),
                        help='Ninapro database version')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory with raw .mat files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--mode', type=str, default='classify',
                        choices=['classify', 'regress'],
                        help='Processing mode')
    parser.add_argument('--window_ms', type=float, default=200.0,
                        help='Window size in milliseconds')
    parser.add_argument('--stride_ms', type=float, default=50.0,
                        help='Stride in milliseconds')
    parser.add_argument('--target_fs', type=float, default=None,
                        help='Target sampling rate (Hz)')
    parser.add_argument('--include_rest', action='store_true',
                        help='Include rest periods')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Fraction of subjects for training')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of subjects for validation')
    
    args = parser.parse_args()
    
    config = DB_CONFIGS[args.db]
    output_dir = os.path.join(args.output_dir, args.db)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all subjects
    all_samples = []
    for subject_id in tqdm(range(1, config['num_subjects'] + 1), desc="Processing subjects"):
        samples = process_subject(
            subject_id=subject_id,
            db_version=args.db,
            root_dir=args.root_dir,
            output_dir=output_dir,
            window_size_ms=args.window_ms,
            stride_ms=args.stride_ms,
            target_fs=args.target_fs,
            mode=args.mode,
            include_rest=args.include_rest,
        )
        all_samples.extend(samples)
    
    print(f"Total samples: {len(all_samples)}")
    
    # Split by subject
    subject_ids = sorted(set(s['subject_id'] for s in all_samples))
    num_subjects = len(subject_ids)
    
    train_cutoff = int(args.train_ratio * num_subjects)
    val_cutoff = int((args.train_ratio + args.val_ratio) * num_subjects)
    
    train_subjects = set(subject_ids[:train_cutoff])
    val_subjects = set(subject_ids[train_cutoff:val_cutoff])
    test_subjects = set(subject_ids[val_cutoff:])
    
    train_samples = [s for s in all_samples if s['subject_id'] in train_subjects]
    val_samples = [s for s in all_samples if s['subject_id'] in val_subjects]
    test_samples = [s for s in all_samples if s['subject_id'] in test_subjects]
    
    print(f"Train: {len(train_samples)} samples from {len(train_subjects)} subjects")
    print(f"Val: {len(val_samples)} samples from {len(val_subjects)} subjects")
    print(f"Test: {len(test_samples)} samples from {len(test_subjects)} subjects")
    
    # Save to HDF5
    save_to_hdf5(train_samples, os.path.join(output_dir, 'train.h5'))
    save_to_hdf5(val_samples, os.path.join(output_dir, 'val.h5'))
    save_to_hdf5(test_samples, os.path.join(output_dir, 'test.h5'))
    
    print(f"Done! Output saved to {output_dir}")


if __name__ == '__main__':
    main()

