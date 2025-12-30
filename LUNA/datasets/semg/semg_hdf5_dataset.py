"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            SEMG_HDF5_DATASET.PY - GENERIC sEMG DATA LOADING                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Flexible PyTorch Dataset for loading sEMG data from HDF5 files.
   Supports classification (gesture labels), regression (pose/kinematics),
   and self-supervised pretraining (no labels).

HIGH-LEVEL OVERVIEW:
   Unlike EEG datasets which have standardized montages, sEMG datasets vary wildly:
   - 4 to 320+ channels
   - Different sampling rates (100-4000 Hz)
   - Labels can be gestures, poses, forces, or nothing (pretraining)
   
   This loader handles all cases with a unified interface.

KEY FEATURES:
   - Multiple task modes: 'pretrain', 'classify', 'regress'
   - Returns dict-style batches (flexible for different heads)
   - Tracks dataset_id and subject_id for multi-dataset training
   - Optional channel subsampling (for channel-agnostic training)
   - Caching for fast repeated access

HDF5 FILE STRUCTURE (expected):
   file.h5
   ├── data_group_0/
   │   ├── X: [N, C, T] sEMG signals (float32)
   │   ├── y: [N] gesture labels (int, optional)
   │   ├── pose: [N, D] joint angles (float32, optional)
   │   ├── subject_id: [N] subject identifiers (int)
   │   └── metadata: group attributes (sampling_rate, dataset_name, etc.)
   └── ...

USAGE MODES:
   1. Pretraining (self-supervised):
      - Returns: {'input': X, 'subject_id': sid, 'num_channels': C, 'dataset_id': did}
   
   2. Classification (gesture recognition):
      - Returns: {'input': X, 'label': y, 'subject_id': sid, ...}
   
   3. Regression (pose estimation):
      - Returns: {'input': X, 'pose': p, 'subject_id': sid, ...}
"""

import torch
import h5py
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any


class sEMGHDF5Dataset(torch.utils.data.Dataset):
    """
    Generic sEMG HDF5 dataset loader supporting multiple task types.
    
    Args:
        hdf5_file: Path to HDF5 file
        mode: One of 'pretrain', 'classify', 'regress'
        dataset_id: Identifier string for this dataset (e.g., 'ninapro_db1')
        num_channels: Expected channel count (for validation/info)
        subsample_channels: If set, randomly subsample to this many channels
        cache_size: Number of samples to cache in memory
        use_cache: Whether to use caching
        normalize: Whether to z-score normalize per channel
        window_size: Expected window size (for validation)
        sampling_rate: Expected sampling rate in Hz (for info)
    """
    
    def __init__(
        self,
        hdf5_file: str,
        mode: str = 'pretrain',  # 'pretrain', 'classify', 'regress'
        dataset_id: str = 'unknown',
        num_channels: Optional[int] = None,
        subsample_channels: Optional[int] = None,
        cache_size: int = 1500,
        use_cache: bool = True,
        normalize: bool = True,
        window_size: Optional[int] = None,
        sampling_rate: float = 2000.0,  # Default, can be overridden
    ):
        super().__init__()
        
        assert mode in ['pretrain', 'classify', 'regress'], \
            f"mode must be 'pretrain', 'classify', or 'regress', got {mode}"
        
        self.hdf5_file = hdf5_file
        self.mode = mode
        self.dataset_id = dataset_id
        self.num_channels = num_channels
        self.subsample_channels = subsample_channels
        self.cache_size = cache_size
        self.use_cache = use_cache
        self.normalize = normalize
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        
        # Open HDF5 file
        self.data = h5py.File(self.hdf5_file, 'r')
        self.keys = list(self.data.keys())
        
        # Build index mapping: flat index → (group_key, sample_idx)
        self.index_map = []
        for key in self.keys:
            if 'X' not in self.data[key]:
                continue  # Skip non-data groups
            group_size = len(self.data[key]['X'])
            self.index_map.extend([(key, i) for i in range(group_size)])
        
        # Infer num_channels from first sample if not provided
        if self.num_channels is None and len(self.index_map) > 0:
            first_key, _ = self.index_map[0]
            self.num_channels = self.data[first_key]['X'].shape[1]
        
        # Initialize cache
        if self.use_cache:
            self.cache: Dict[int, Any] = {}
            self.cache_queue = deque(maxlen=self.cache_size)
        
        # Check what fields are available
        if len(self.keys) > 0:
            sample_grp = self.data[self.keys[0]]
            self.has_labels = 'y' in sample_grp
            self.has_pose = 'pose' in sample_grp
            self.has_subject_id = 'subject_id' in sample_grp
        else:
            self.has_labels = False
            self.has_pose = False
            self.has_subject_id = False
        
        # Validate mode vs available data
        if mode == 'classify' and not self.has_labels:
            print(f"Warning: mode='classify' but no 'y' field in {hdf5_file}")
        if mode == 'regress' and not self.has_pose:
            print(f"Warning: mode='regress' but no 'pose' field in {hdf5_file}")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def _normalize_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalize per channel."""
        # x: [C, T]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std
    
    def _subsample_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly subsample channels to simulate lower-density hardware."""
        if self.subsample_channels is None or x.shape[0] <= self.subsample_channels:
            return x
        
        # Random channel selection (consistent within epoch via torch.randperm)
        indices = torch.randperm(x.shape[0])[:self.subsample_channels]
        indices = indices.sort().values  # Keep spatial order
        return x[indices]
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Check cache first
        if self.use_cache and index in self.cache:
            return self.cache[index]
        
        # Load from HDF5
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        
        # Load EMG signal
        X = grp['X'][sample_idx]
        X = torch.FloatTensor(X)  # [C, T]
        
        # Normalize
        if self.normalize:
            X = self._normalize_signal(X)
        
        # Subsample channels
        X = self._subsample_channels(X)
        
        # Build output dict
        output = {
            'input': X,
            'num_channels': X.shape[0],
            'dataset_id': self.dataset_id,
        }
        
        # Add subject_id if available
        if self.has_subject_id:
            output['subject_id'] = int(grp['subject_id'][sample_idx])
        else:
            output['subject_id'] = 0  # Default
        
        # Add task-specific fields
        if self.mode == 'classify' and self.has_labels:
            y = grp['y'][sample_idx]
            output['label'] = torch.LongTensor([y]).squeeze()
        
        if self.mode == 'regress' and self.has_pose:
            pose = grp['pose'][sample_idx]
            output['pose'] = torch.FloatTensor(pose)
        
        # Cache the result
        if self.use_cache:
            self.cache[index] = output
            self.cache_queue.append(index)
            
            # Evict old entries if cache is full
            if len(self.cache) > self.cache_size:
                old_idx = self.cache_queue.popleft()
                if old_idx in self.cache:
                    del self.cache[old_idx]
        
        return output
    
    def read_info(self, index: int) -> Dict[str, Any]:
        """
        Read metadata for a sample without loading full signal.
        Used for subject-independent splitting.
        """
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        
        info = {'index': index}
        
        if self.has_subject_id:
            info['subject_id'] = int(grp['subject_id'][sample_idx])
            info['trial_id'] = info['subject_id']  # Alias for compatibility
        
        return info
    
    @property
    def train_size(self) -> int:
        """For compatibility with SubjectIndependentDataModule."""
        # Default: 60% of subjects for training
        if not self.has_subject_id:
            return int(0.6 * len(self))
        
        # Get unique subject IDs
        subject_ids = set()
        for i in range(min(1000, len(self))):  # Sample first 1000
            info = self.read_info(i)
            subject_ids.add(info['subject_id'])
        
        return int(0.6 * max(subject_ids))
    
    @property
    def val_size(self) -> int:
        """For compatibility with SubjectIndependentDataModule."""
        if not self.has_subject_id:
            return int(0.2 * len(self))
        
        subject_ids = set()
        for i in range(min(1000, len(self))):
            info = self.read_info(i)
            subject_ids.add(info['subject_id'])
        
        return int(0.2 * max(subject_ids))
    
    @property
    def test_size(self) -> int:
        """For compatibility with SubjectIndependentDataModule."""
        if not self.has_subject_id:
            return int(0.2 * len(self))
        
        subject_ids = set()
        for i in range(min(1000, len(self))):
            info = self.read_info(i)
            subject_ids.add(info['subject_id'])
        
        return int(0.2 * max(subject_ids))
    
    def __del__(self):
        """Close HDF5 file on deletion."""
        if hasattr(self, 'data') and self.data:
            self.data.close()
    
    def __repr__(self) -> str:
        return (
            f"sEMGHDF5Dataset(\n"
            f"  file={self.hdf5_file},\n"
            f"  mode={self.mode},\n"
            f"  dataset_id={self.dataset_id},\n"
            f"  num_samples={len(self)},\n"
            f"  num_channels={self.num_channels},\n"
            f"  has_labels={self.has_labels},\n"
            f"  has_pose={self.has_pose},\n"
            f")"
        )

