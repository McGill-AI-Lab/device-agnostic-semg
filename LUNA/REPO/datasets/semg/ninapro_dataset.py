"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NINAPRO_DATASET.PY - NINAPRO sEMG DATABASE LOADER               ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Dataset loaders for the Ninapro (Non-Invasive Adaptive Prosthetics) database.
   The most widely used public sEMG dataset family, with multiple sub-databases.

NINAPRO DATABASE OVERVIEW:
   
   Classification-focused (gesture labels only):
   ┌──────────┬────────┬──────────┬─────────┬──────────────────────────────────┐
   │ Database │ Ch     │ Subjects │ Gestures│ Hardware                         │
   ├──────────┼────────┼──────────┼─────────┼──────────────────────────────────┤
   │ DB1      │ 10     │ 27       │ 52      │ Otto Bock MyoBock                │
   │ DB2      │ 12     │ 40       │ 49      │ Delsys Trigno Wireless           │
   │ DB3      │ 12     │ 11       │ 49      │ Delsys (amputees)                │
   │ DB4      │ 12     │ 10       │ 52      │ Cometa (2 kHz)                   │
   │ DB5      │ 16     │ 10       │ 52      │ 2x Myo armband (200 Hz)          │
   │ DB6      │ 14     │ 10       │ 7       │ Delsys (5 days multi-session)    │
   └──────────┴────────┴──────────┴─────────┴──────────────────────────────────┘
   
   Pose-capable (have CyberGlove data for regression):
   ┌──────────┬────────┬─────────┬──────────────────────────────────────────────┐
   │ Database │ Ch     │ DOF     │ Description                                  │
   ├──────────┼────────┼─────────┼──────────────────────────────────────────────┤
   │ DB1      │ 10     │ 22      │ CyberGlove II + fingertip force sensors      │
   │ DB2      │ 12     │ 22      │ CyberGlove II                                │
   │ DB4      │ 12     │ 22      │ CyberGlove II, high sampling rate            │
   │ DB5      │ 16     │ 22      │ CyberGlove II, Myo hardware                  │
   │ DB7      │ 12     │ 18      │ Delsys + IMU, 18-DOF glove                   │
   │ DB8      │ 16     │ 18      │ Designed specifically for finger regression  │
   └──────────┴────────┴─────────┴──────────────────────────────────────────────┘

KEY CLASSES:
   
   NinaproDataset: For gesture classification tasks
   - Returns: {'input': emg, 'label': gesture_id, 'subject_id': sid, ...}
   
   NinaproPoseDataset: For pose regression tasks (uses glove DOFs)
   - Returns: {'input': emg, 'pose': joint_angles, 'subject_id': sid, ...}

DATA FORMAT:
   Ninapro provides .mat files with:
   - emg: [T, C] EMG signals (we transpose to [C, T])
   - stimulus: [T, 1] gesture labels during movement
   - restimulus: [T, 1] gesture labels including repetitions
   - glove: [T, 22] CyberGlove joint angles (if available)
   - subject: subject ID
   - exercise: exercise set (1, 2, or 3)

PREPROCESSING NOTES:
   - Original signals are continuous; we window them
   - Gesture transitions need careful handling
   - Rest periods (label 0) can be included or excluded
   - Different DBs have different sampling rates (100-2000 Hz)
"""

import os
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

try:
    import scipy.io as sio
except ImportError:
    sio = None


# Ninapro database configurations
NINAPRO_CONFIGS = {
    'db1': {
        'num_channels': 10,
        'num_gestures': 52,
        'num_subjects': 27,
        'sampling_rate': 100,
        'has_glove': True,
        'glove_dof': 22,
        'hardware': 'Otto Bock MyoBock',
    },
    'db2': {
        'num_channels': 12,
        'num_gestures': 49,
        'num_subjects': 40,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_dof': 22,
        'hardware': 'Delsys Trigno Wireless',
    },
    'db3': {
        'num_channels': 12,
        'num_gestures': 49,
        'num_subjects': 11,
        'sampling_rate': 2000,
        'has_glove': False,
        'glove_dof': 0,
        'hardware': 'Delsys (amputees)',
        'is_amputee': True,
    },
    'db4': {
        'num_channels': 12,
        'num_gestures': 52,
        'num_subjects': 10,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_dof': 22,
        'hardware': 'Cometa',
    },
    'db5': {
        'num_channels': 16,
        'num_gestures': 52,
        'num_subjects': 10,
        'sampling_rate': 200,
        'has_glove': True,
        'glove_dof': 22,
        'hardware': '2x Myo armband',
    },
    'db6': {
        'num_channels': 14,
        'num_gestures': 7,
        'num_subjects': 10,
        'sampling_rate': 2000,
        'has_glove': False,
        'glove_dof': 0,
        'hardware': 'Delsys (multi-day)',
        'num_days': 5,
        'sessions_per_day': 2,
    },
    'db7': {
        'num_channels': 12,
        'num_gestures': 40,
        'num_subjects': 20,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_dof': 18,
        'hardware': 'Delsys Trigno + IMU',
    },
    'db8': {
        'num_channels': 16,
        'num_gestures': 9,
        'num_subjects': 10,
        'sampling_rate': 2000,
        'has_glove': True,
        'glove_dof': 18,
        'hardware': 'Delsys Trigno',
        'designed_for': 'finger regression',
    },
}


class NinaproDataset(torch.utils.data.Dataset):
    """
    Ninapro dataset for gesture classification.
    
    Loads preprocessed HDF5 files (after running preprocessing script).
    Each sample is a windowed EMG segment with gesture label.
    
    Args:
        hdf5_file: Path to preprocessed HDF5 file
        db_version: Ninapro database version ('db1', 'db2', ..., 'db8')
        include_rest: Include rest periods (gesture 0) in dataset
        normalize: Z-score normalize EMG per channel
        window_size: Window size in samples (if None, use default for DB)
        subsample_channels: Randomly subsample to N channels (for training)
    """
    
    def __init__(
        self,
        hdf5_file: str,
        db_version: str = 'db2',
        include_rest: bool = False,
        normalize: bool = True,
        window_size: Optional[int] = None,
        subsample_channels: Optional[int] = None,
    ):
        super().__init__()
        
        assert db_version in NINAPRO_CONFIGS, \
            f"Unknown DB version: {db_version}. Choose from {list(NINAPRO_CONFIGS.keys())}"
        
        self.hdf5_file = hdf5_file
        self.db_version = db_version
        self.config = NINAPRO_CONFIGS[db_version]
        self.include_rest = include_rest
        self.normalize = normalize
        self.window_size = window_size
        self.subsample_channels = subsample_channels
        
        self.num_channels = self.config['num_channels']
        self.num_gestures = self.config['num_gestures']
        self.sampling_rate = self.config['sampling_rate']
        
        # Load HDF5 file
        import h5py
        self.data = h5py.File(self.hdf5_file, 'r')
        
        # Build index
        self.index_map = []
        for key in self.data.keys():
            if 'X' not in self.data[key]:
                continue
            
            group_size = len(self.data[key]['X'])
            for i in range(group_size):
                # Optionally filter rest
                if not include_rest and 'y' in self.data[key]:
                    if self.data[key]['y'][i] == 0:
                        continue
                
                self.index_map.append((key, i))
        
        # For subject-independent splits
        self._scan_subjects()
    
    def _scan_subjects(self, max_scan: int = 5000):
        """Scan dataset to find unique subjects."""
        self.subject_ids = set()
        for i in range(min(max_scan, len(self.index_map))):
            group_key, sample_idx = self.index_map[i]
            if 'subject_id' in self.data[group_key]:
                sid = int(self.data[group_key]['subject_id'][sample_idx])
                self.subject_ids.add(sid)
        
        self.subject_ids = sorted(self.subject_ids)
        self._num_subjects = len(self.subject_ids) if self.subject_ids else self.config['num_subjects']
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        
        # Load EMG
        X = torch.FloatTensor(grp['X'][sample_idx])  # [C, T]
        
        # Normalize
        if self.normalize:
            mean = X.mean(dim=1, keepdim=True)
            std = X.std(dim=1, keepdim=True) + 1e-8
            X = (X - mean) / std
        
        # Subsample channels
        if self.subsample_channels and X.shape[0] > self.subsample_channels:
            indices = torch.randperm(X.shape[0])[:self.subsample_channels].sort().values
            X = X[indices]
        
        # Load label
        y = grp['y'][sample_idx]
        y = torch.LongTensor([y]).squeeze()
        
        # Subject ID
        if 'subject_id' in grp:
            subject_id = int(grp['subject_id'][sample_idx])
        else:
            subject_id = 0
        
        return {
            'input': X,
            'label': y,
            'subject_id': subject_id,
            'num_channels': X.shape[0],
            'dataset_id': f'ninapro_{self.db_version}',
        }
    
    def read_info(self, index: int) -> Dict[str, Any]:
        """Read sample metadata for splitting."""
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        
        info = {'index': index}
        if 'subject_id' in grp:
            info['subject_id'] = int(grp['subject_id'][sample_idx])
            info['trial_id'] = info['subject_id']
        else:
            info['subject_id'] = 0
            info['trial_id'] = 0
        
        return info
    
    @property
    def train_size(self) -> int:
        return int(0.6 * self._num_subjects)
    
    @property
    def val_size(self) -> int:
        return int(0.2 * self._num_subjects)
    
    @property
    def test_size(self) -> int:
        return self._num_subjects - self.train_size - self.val_size
    
    def __del__(self):
        if hasattr(self, 'data') and self.data:
            self.data.close()


class NinaproPoseDataset(torch.utils.data.Dataset):
    """
    Ninapro dataset for pose regression (using CyberGlove data).
    
    Only works with DB1, DB2, DB4, DB5, DB7, DB8 which have glove data.
    Returns continuous joint angles instead of discrete gesture labels.
    
    Args:
        hdf5_file: Path to preprocessed HDF5 file with pose data
        db_version: Ninapro database version
        target_dof: Number of DOF to predict (can subset the full glove DOF)
        normalize_pose: Normalize pose to [-1, 1] range
        normalize_emg: Z-score normalize EMG per channel
    """
    
    def __init__(
        self,
        hdf5_file: str,
        db_version: str = 'db2',
        target_dof: Optional[int] = None,
        normalize_pose: bool = True,
        normalize_emg: bool = True,
        subsample_channels: Optional[int] = None,
    ):
        super().__init__()
        
        assert db_version in NINAPRO_CONFIGS, f"Unknown DB: {db_version}"
        
        config = NINAPRO_CONFIGS[db_version]
        assert config['has_glove'], f"{db_version} does not have glove data"
        
        self.hdf5_file = hdf5_file
        self.db_version = db_version
        self.config = config
        self.target_dof = target_dof or config['glove_dof']
        self.normalize_pose = normalize_pose
        self.normalize_emg = normalize_emg
        self.subsample_channels = subsample_channels
        
        self.num_channels = config['num_channels']
        self.glove_dof = config['glove_dof']
        
        # Load HDF5
        import h5py
        self.data = h5py.File(self.hdf5_file, 'r')
        
        # Build index (only include samples with valid pose)
        self.index_map = []
        for key in self.data.keys():
            if 'X' not in self.data[key] or 'pose' not in self.data[key]:
                continue
            
            group_size = len(self.data[key]['X'])
            self.index_map.extend([(key, i) for i in range(group_size)])
        
        self._scan_subjects()
    
    def _scan_subjects(self, max_scan: int = 5000):
        self.subject_ids = set()
        for i in range(min(max_scan, len(self.index_map))):
            group_key, sample_idx = self.index_map[i]
            if 'subject_id' in self.data[group_key]:
                sid = int(self.data[group_key]['subject_id'][sample_idx])
                self.subject_ids.add(sid)
        
        self.subject_ids = sorted(self.subject_ids)
        self._num_subjects = len(self.subject_ids) if self.subject_ids else self.config['num_subjects']
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        
        # Load EMG
        X = torch.FloatTensor(grp['X'][sample_idx])
        
        if self.normalize_emg:
            mean = X.mean(dim=1, keepdim=True)
            std = X.std(dim=1, keepdim=True) + 1e-8
            X = (X - mean) / std
        
        if self.subsample_channels and X.shape[0] > self.subsample_channels:
            indices = torch.randperm(X.shape[0])[:self.subsample_channels].sort().values
            X = X[indices]
        
        # Load pose (joint angles)
        pose = torch.FloatTensor(grp['pose'][sample_idx])
        
        # Subset DOF if requested
        if self.target_dof < pose.shape[0]:
            pose = pose[:self.target_dof]
        
        if self.normalize_pose:
            # Normalize to approximately [-1, 1] range
            pose = pose / 180.0  # Assuming degrees, rough normalization
        
        # Subject ID
        subject_id = 0
        if 'subject_id' in grp:
            subject_id = int(grp['subject_id'][sample_idx])
        
        return {
            'input': X,
            'pose': pose,
            'subject_id': subject_id,
            'num_channels': X.shape[0],
            'dataset_id': f'ninapro_{self.db_version}_pose',
        }
    
    def read_info(self, index: int) -> Dict[str, Any]:
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        
        info = {'index': index}
        if 'subject_id' in grp:
            info['subject_id'] = int(grp['subject_id'][sample_idx])
            info['trial_id'] = info['subject_id']
        else:
            info['subject_id'] = 0
            info['trial_id'] = 0
        
        return info
    
    @property
    def train_size(self) -> int:
        return int(0.6 * self._num_subjects)
    
    @property
    def val_size(self) -> int:
        return int(0.2 * self._num_subjects)
    
    @property
    def test_size(self) -> int:
        return self._num_subjects - self.train_size - self.val_size
    
    def __del__(self):
        if hasattr(self, 'data') and self.data:
            self.data.close()

# Utility functions for raw Ninapro processing

def get_ninapro_info(db_version: str) -> Dict[str, Any]:
    """Get configuration info for a Ninapro database."""
    return NINAPRO_CONFIGS.get(db_version, {})


def list_available_dbs() -> List[str]:
    """List all supported Ninapro databases."""
    return list(NINAPRO_CONFIGS.keys())

