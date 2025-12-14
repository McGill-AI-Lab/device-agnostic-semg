"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SUBJECT_INDEPENDENT_DATA_MODULE.PY - CROSS-SUBJECT SPLITS           ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   DataModule for subject-independent evaluation where train/val/test are split by
   subject/trial ID rather than random sampling. Ensures no subject leakage across splits.

HIGH-LEVEL OVERVIEW:
   In many EEG experiments (e.g., SEED-V emotion recognition), proper evaluation requires
   testing on unseen subjects. This module:
   1. Reads trial_id from each sample's metadata
   2. Assigns samples to train/val/test based on trial_id thresholds
   3. Ensures subject-independent splits (no data leakage)
   4. Concatenates multiple datasets while preserving split boundaries

KEY CLASSES:
   
   SubjectIndependentDataModule(pl.LightningDataModule):
   - Requires datasets with read_info() method returning trial_id
   - Splits based on trial_id ranges (not random sampling)
   - Supports datasets with or without explicit test sets
   - Concatenates all train splits, val splits, test splits separately

SPLITTING LOGIC:
   For each dataset:
   1. Read train_size, val_size, test_size from dataset
   2. For each sample, check its trial_id:
      - trial_id <= train_size → training set
      - train_size < trial_id <= train_size + val_size → validation set
      - trial_id > train_size + val_size → test set
   3. If test_size == 0: use validation set for testing
   
   This ensures subjects in training never appear in val/test.

TYPICAL USAGE:
   Used for:
   - SEED-V emotion recognition (subject-independent protocol)
   - Any task requiring cross-subject generalization
   - Datasets where subjects/trials are explicitly indexed
   
   Not used for:
   - Random train/val/test splits (use FinetuneDataModule)
   - Simple time-series splits (use PretrainDataModule)

RELATED FILES:
   - datasets/seed_v_dataset.py: Implements trial_id-based indexing
   - tasks/finetune_task_LUNA.py: Can use this for evaluation
   - config/data_module/subject_independent_data_module.yaml: Configuration
"""
#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Berkay Döner                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import (
    ConcatDataset
)
import torch

class SubjectIndependentDataModule(pl.LightningDataModule):

    def __init__(self, datasets, cfg=None, name="", split_dir=None, **kwargs):
        super().__init__()
        self.train, self.val, self.test = [], [], []
        datasets_list = [
            datasets[dataset_name]
            for dataset_name in datasets
            if datasets[dataset_name] is not None
        ]
        for dataset in datasets_list:
            train_size, val_size, test_size = int(dataset.train_size), int(dataset.val_size), int(dataset.test_size)
            train_ids, val_ids, test_ids = [], [], []
            if test_size == 0:
                for i in range(len(dataset)):
                    if dataset.read_info(i)['trial_id'] <= train_size: 
                        train_ids.append(i)
                    else:
                        val_ids.append(i)
                train = torch.utils.data.Subset(dataset, train_ids)
                val = torch.utils.data.Subset(dataset, val_ids)
                self.train.append(train)
                self.val.append(val)
                self.test.append(val)
            else:
                for i in range(len(dataset)):
                    if dataset.read_info(i)['trial_id'] <= train_size:
                        train_ids.append(i)
                    elif dataset.read_info(i)['trial_id'] <= train_size + val_size:
                        val_ids.append(i)
                    else:
                        test_ids.append(i)
                train = torch.utils.data.Subset(dataset, train_ids)
                val = torch.utils.data.Subset(dataset, val_ids)
                test = torch.utils.data.Subset(dataset, test_ids)
                self.train.append(train)
                self.val.append(val)
                self.test.append(test)
        self.train = ConcatDataset(self.train)
        self.val = ConcatDataset(self.val)
        self.test = ConcatDataset(self.test)
        self.name = name
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.train
            self.val_dataset = self.val

        # Assign test dataset for use in dataloader(s)
        elif stage == "validate":
            self.val_dataset = self.val
        elif stage == "test":
            self.test_dataset = self.test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )