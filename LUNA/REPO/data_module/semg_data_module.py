"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SEMG_DATA_MODULE.PY - sEMG MULTI-DATASET LOADING               ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   PyTorch Lightning DataModule for sEMG experiments supporting:
   - Multiple datasets with varying channel counts
   - Both classification (gesture) and regression (pose) tasks
   - Subject-independent train/val/test splits
   - Channel-count grouping for efficient batching

HIGH-LEVEL OVERVIEW:
   sEMG datasets vary widely in channel count (4-320), sampling rate, and labels.
   This module provides a unified interface for:
   1. Loading multiple sEMG datasets together
   2. Splitting by subject (no data leakage)
   3. Grouping by channel count (for LUNA's channel-agnostic training)
   4. Handling both gesture classification and pose regression

KEY CLASSES:
   
   sEMGDataModule(pl.LightningDataModule):
   - General purpose sEMG data module
   - Single dataset or multiple datasets
   - Subject-independent splitting
   
   MultiDatasetEMGModule(pl.LightningDataModule):
   - Combines multiple sEMG datasets
   - Groups by channel count for efficient batching
   - Supports sequential or interleaved loading

COLLATE FUNCTIONS:
   - semg_collate_fn: Handles dict-style samples
   - semg_collate_padded: Pads variable-length channels to max

USAGE:
   # Single dataset
   dm = sEMGDataModule(
       train_dataset=NinaproDataset('train.h5', db_version='db2'),
       val_dataset=NinaproDataset('val.h5', db_version='db2'),
       batch_size=64,
   )
   
   # Multi-dataset with channel grouping
   dm = MultiDatasetEMGModule(
       datasets={
           'db2': NinaproDataset('db2.h5', db_version='db2'),  # 12 ch
           'db5': NinaproDataset('db5.h5', db_version='db5'),  # 16 ch
       },
       batch_size=64,
       group_by_channels=True,
   )
"""

from typing import Optional, Dict, List, Any, Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import torch
import numpy as np
from collections import defaultdict


# ============================================================================
# Collate Functions for sEMG Dict Samples
# ============================================================================

def semg_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for dict-style sEMG samples.
    
    Handles samples like:
    {
        'input': tensor [C, T],
        'label': tensor (optional),
        'pose': tensor [D] (optional),
        'subject_id': int,
        'num_channels': int,
        'dataset_id': str,
    }
    
    Returns batched dict with stacked tensors.
    """
    # Stack inputs (assumes same channel count in batch)
    inputs = torch.stack([s['input'] for s in batch])
    
    result = {
        'input': inputs,
        'num_channels': batch[0]['num_channels'],
        'dataset_id': batch[0]['dataset_id'],
    }
    
    # Handle subject IDs
    if 'subject_id' in batch[0]:
        result['subject_id'] = torch.LongTensor([s['subject_id'] for s in batch])
    
    # Handle labels (classification)
    if 'label' in batch[0]:
        result['label'] = torch.stack([s['label'] for s in batch])
    
    # Handle pose (regression)
    if 'pose' in batch[0]:
        result['pose'] = torch.stack([s['pose'] for s in batch])
    
    return result


def semg_collate_padded(batch: List[Dict[str, Any]], 
                        pad_channels: int = None) -> Dict[str, Any]:
    """
    Collate function that pads channels to uniform size.
    
    Useful when batching samples with different channel counts.
    Pads smaller signals with zeros and creates a channel mask.
    
    Args:
        batch: List of samples
        pad_channels: Target channel count (None = use max in batch)
    """
    # Find max channels in batch
    max_channels = max(s['input'].shape[0] for s in batch)
    if pad_channels is not None:
        max_channels = max(max_channels, pad_channels)
    
    # Get time dimension
    T = batch[0]['input'].shape[1]
    
    # Pad and stack
    padded_inputs = []
    channel_masks = []
    
    for s in batch:
        C = s['input'].shape[0]
        if C < max_channels:
            # Pad with zeros
            padding = torch.zeros(max_channels - C, T)
            padded = torch.cat([s['input'], padding], dim=0)
            mask = torch.cat([torch.ones(C), torch.zeros(max_channels - C)])
        else:
            padded = s['input']
            mask = torch.ones(C)
        
        padded_inputs.append(padded)
        channel_masks.append(mask)
    
    result = {
        'input': torch.stack(padded_inputs),
        'channel_mask': torch.stack(channel_masks),  # [B, C] binary mask
        'original_channels': torch.LongTensor([s['num_channels'] for s in batch]),
    }
    
    # Copy other fields
    if 'subject_id' in batch[0]:
        result['subject_id'] = torch.LongTensor([s['subject_id'] for s in batch])
    
    if 'label' in batch[0]:
        result['label'] = torch.stack([s['label'] for s in batch])
    
    if 'pose' in batch[0]:
        result['pose'] = torch.stack([s['pose'] for s in batch])
    
    result['dataset_id'] = batch[0].get('dataset_id', 'unknown')
    
    return result


# ============================================================================
# Single Dataset Data Module
# ============================================================================

class sEMGDataModule(pl.LightningDataModule):
    """
    Data module for a single sEMG dataset or pre-split train/val/test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        batch_size: Batch size for all dataloaders
        num_workers: Number of dataloader workers
        pin_memory: Use pinned memory for GPU transfer
        collate_fn: Custom collate function (default: semg_collate_fn)
    """
    
    def __init__(
        self,
        train_dataset: Dataset = None,
        val_dataset: Dataset = None,
        test_dataset: Dataset = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: callable = None,
        cfg: Any = None,  # For compatibility with Hydra configs
        **kwargs,
    ):
        super().__init__()
        
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset or val_dataset
        
        # Support both direct args and cfg object
        if cfg is not None:
            self.batch_size = getattr(cfg, 'batch_size', batch_size)
            self.num_workers = getattr(cfg, 'num_workers', num_workers)
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers
        
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn or semg_collate_fn
    
    def setup(self, stage: Optional[str] = None):
        """Setup is called once per process."""
        if stage == 'fit' or stage is None:
            self.train_dataset = self.train_ds
            self.val_dataset = self.val_ds
        
        if stage == 'validate':
            self.val_dataset = self.val_ds
        
        if stage == 'test':
            self.test_dataset = self.test_ds
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn,
        )


# ============================================================================
# Multi-Dataset Data Module (Groups by Channel Count)
# ============================================================================

class SequentialMultiLoader:
    """
    Chains multiple dataloaders, iterating through them sequentially.
    Used when grouping by channel count.
    """
    
    def __init__(self, dataloaders: List[DataLoader]):
        self.dataloaders = dataloaders
    
    def __len__(self) -> int:
        return sum(len(dl) for dl in self.dataloaders)
    
    def __iter__(self):
        for dl in self.dataloaders:
            yield from dl


class MultiDatasetEMGModule(pl.LightningDataModule):
    """
    Data module that combines multiple sEMG datasets with varying channels.
    
    Supports two modes:
    1. group_by_channels=True: Groups datasets by channel count, creates
       separate loaders per group, iterates sequentially. Efficient for
       channel-agnostic models like LUNA.
    
    2. group_by_channels=False: Concatenates all datasets, pads to max
       channels, includes channel mask. Less efficient but simpler.
    
    Args:
        datasets: Dict mapping dataset_id to Dataset objects
        batch_size: Batch size
        num_workers: Number of workers
        group_by_channels: Whether to group by channel count
        train_val_split: Ratio for automatic train/val split (if not pre-split)
        cfg: Hydra config object
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int = 64,
        num_workers: int = 4,
        group_by_channels: bool = True,
        train_val_split: float = 0.8,
        cfg: Any = None,
        **kwargs,
    ):
        super().__init__()
        
        self.raw_datasets = datasets
        self.group_by_channels = group_by_channels
        self.train_val_split = train_val_split
        
        if cfg is not None:
            self.batch_size = getattr(cfg, 'batch_size', batch_size)
            self.num_workers = getattr(cfg, 'num_workers', num_workers)
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers
        
        # Process datasets
        self._prepare_datasets()
    
    def _get_num_channels(self, dataset: Dataset) -> int:
        """Get channel count from a dataset."""
        if hasattr(dataset, 'num_channels'):
            return dataset.num_channels
        
        # Try to infer from first sample
        try:
            sample = dataset[0]
            if isinstance(sample, dict):
                return sample['input'].shape[0]
            elif isinstance(sample, tuple):
                return sample[0].shape[0]
            else:
                return sample.shape[0]
        except:
            return -1  # Unknown
    
    def _prepare_datasets(self):
        """Organize datasets by channel count and split."""
        
        # Filter out None datasets
        datasets = {k: v for k, v in self.raw_datasets.items() if v is not None}
        
        if self.group_by_channels:
            # Group by channel count
            channel_groups: Dict[int, List[Dataset]] = defaultdict(list)
            
            for name, ds in datasets.items():
                num_ch = self._get_num_channels(ds)
                channel_groups[num_ch].append(ds)
            
            # Create train/val splits per group
            self.train_groups: Dict[int, ConcatDataset] = {}
            self.val_groups: Dict[int, ConcatDataset] = {}
            
            for num_ch, group_datasets in channel_groups.items():
                train_list, val_list = [], []
                
                for ds in group_datasets:
                    n = len(ds)
                    train_n = int(self.train_val_split * n)
                    indices = torch.randperm(n).tolist()
                    
                    train_list.append(Subset(ds, indices[:train_n]))
                    val_list.append(Subset(ds, indices[train_n:]))
                
                self.train_groups[num_ch] = ConcatDataset(train_list)
                self.val_groups[num_ch] = ConcatDataset(val_list)
            
            print(f"Created {len(self.train_groups)} channel groups: {list(self.train_groups.keys())}")
        
        else:
            # Simple concatenation (will pad in collate)
            all_datasets = list(datasets.values())
            train_list, val_list = [], []
            
            for ds in all_datasets:
                n = len(ds)
                train_n = int(self.train_val_split * n)
                indices = torch.randperm(n).tolist()
                
                train_list.append(Subset(ds, indices[:train_n]))
                val_list.append(Subset(ds, indices[train_n:]))
            
            self.train_concat = ConcatDataset(train_list)
            self.val_concat = ConcatDataset(val_list)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for the given stage."""
        pass  # Already prepared in __init__
    
    def train_dataloader(self) -> Union[DataLoader, SequentialMultiLoader]:
        if self.group_by_channels:
            # Create loader per channel group
            loaders = []
            for num_ch, ds in sorted(self.train_groups.items()):
                loader = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=semg_collate_fn,
                )
                loaders.append(loader)
            
            return SequentialMultiLoader(loaders)
        
        else:
            return DataLoader(
                self.train_concat,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=semg_collate_padded,
            )
    
    def val_dataloader(self) -> Union[DataLoader, SequentialMultiLoader]:
        if self.group_by_channels:
            loaders = []
            for num_ch, ds in sorted(self.val_groups.items()):
                loader = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=semg_collate_fn,
                )
                loaders.append(loader)
            
            return SequentialMultiLoader(loaders)
        
        else:
            return DataLoader(
                self.val_concat,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=semg_collate_padded,
            )


# ============================================================================
# Subject-Independent sEMG Data Module
# ============================================================================

class SubjectIndependentEMGModule(pl.LightningDataModule):
    """
    Data module with subject-independent train/val/test splits.
    
    Ensures no subject appears in multiple splits to prevent data leakage.
    Particularly important for sEMG where subjects have unique muscle patterns.
    
    Args:
        dataset: Full dataset with subject_id information
        train_subjects_ratio: Fraction of subjects for training
        val_subjects_ratio: Fraction of subjects for validation
        batch_size: Batch size
        num_workers: Number of workers
        cfg: Hydra config object
    """
    
    def __init__(
        self,
        dataset: Dataset,
        train_subjects_ratio: float = 0.6,
        val_subjects_ratio: float = 0.2,
        batch_size: int = 64,
        num_workers: int = 4,
        cfg: Any = None,
        **kwargs,
    ):
        super().__init__()
        
        self.dataset = dataset
        self.train_ratio = train_subjects_ratio
        self.val_ratio = val_subjects_ratio
        
        if cfg is not None:
            self.batch_size = getattr(cfg, 'batch_size', batch_size)
            self.num_workers = getattr(cfg, 'num_workers', num_workers)
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers
        
        self._split_by_subject()
    
    def _split_by_subject(self):
        """Create subject-independent splits."""
        
        # Collect subject IDs for each sample
        subject_to_indices: Dict[int, List[int]] = defaultdict(list)
        
        for i in range(len(self.dataset)):
            if hasattr(self.dataset, 'read_info'):
                info = self.dataset.read_info(i)
                sid = info.get('subject_id', 0)
            else:
                sample = self.dataset[i]
                if isinstance(sample, dict):
                    sid = sample.get('subject_id', 0)
                else:
                    sid = 0
            
            subject_to_indices[sid].append(i)
        
        # Split subjects
        subjects = sorted(subject_to_indices.keys())
        n_subjects = len(subjects)
        
        train_cutoff = int(self.train_ratio * n_subjects)
        val_cutoff = int((self.train_ratio + self.val_ratio) * n_subjects)
        
        train_subjects = set(subjects[:train_cutoff])
        val_subjects = set(subjects[train_cutoff:val_cutoff])
        test_subjects = set(subjects[val_cutoff:])
        
        # Collect indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        for sid, indices in subject_to_indices.items():
            if sid in train_subjects:
                train_indices.extend(indices)
            elif sid in val_subjects:
                val_indices.extend(indices)
            else:
                test_indices.extend(indices)
        
        self.train_ds = Subset(self.dataset, train_indices)
        self.val_ds = Subset(self.dataset, val_indices)
        self.test_ds = Subset(self.dataset, test_indices)
        
        print(f"Subject-independent split:")
        print(f"  Train: {len(train_subjects)} subjects, {len(train_indices)} samples")
        print(f"  Val: {len(val_subjects)} subjects, {len(val_indices)} samples")
        print(f"  Test: {len(test_subjects)} subjects, {len(test_indices)} samples")
    
    def setup(self, stage: Optional[str] = None):
        pass  # Already split in __init__
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=semg_collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=semg_collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=semg_collate_fn,
        )

