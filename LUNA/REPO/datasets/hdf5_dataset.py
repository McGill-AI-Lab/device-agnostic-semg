"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 HDF5_DATASET.PY - EFFICIENT EEG DATA LOADING                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   PyTorch Dataset for loading EEG data from HDF5 files with intelligent caching
   for fast I/O. Supports both pretraining (no labels) and finetuning (with labels).

HIGH-LEVEL OVERVIEW:
   HDF5 format enables efficient storage and random access for large EEG datasets.
   This loader:
   1. Opens HDF5 file with hierarchical group structure
   2. Builds index mapping for O(1) sample access
   3. Implements LRU-style cache to reduce disk reads
   4. Returns either (X) for pretraining or (X, y) for finetuning

KEY CLASSES:
   
   HDF5Loader(torch.utils.data.Dataset):
   - Opens HDF5 file and maintains connection
   - Maps flat indices to (group_key, sample_idx) pairs
   - Caches recently accessed samples in memory
   - Converts to PyTorch tensors on-the-fly

KEY FEATURES:
   - Caching: Stores up to cache_size samples (default 1500) in memory
   - Lazy loading: Only loads samples when accessed
   - Group-based storage: Samples organized into groups for efficiency
   - Flexible modes:
     * finetune=True: Returns (X, y) for supervised learning
     * finetune=False: Returns X for self-supervised learning
   - Optional squeeze: Adds channel dimension if needed

HDF5 FILE STRUCTURE:
   file.h5
   ├── data_group_0/
   │   ├── X: [N, C, T] EEG signals
   │   └── y: [N] labels (optional)
   ├── data_group_1/
   │   ├── X: [N, C, T]
   │   └── y: [N]
   └── ...
   
   Where:
   - N = samples per group (typically 1000)
   - C = number of EEG channels
   - T = time samples (e.g., 5120 for 20s at 256Hz)

RELATED FILES:
   - make_datasets/make_hdf5.py: Creates HDF5 files from pickles
   - data_module/*: Use this dataset in their dataloaders
   - datasets/seed_v_dataset.py: Alternative dataset for SEED-V
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
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
import h5py
from collections import deque

class HDF5Loader(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, squeeze=False, finetune=True, cache_size=1500, use_cache=True):
        self.hdf5_file = hdf5_file
        self.squeeze = squeeze
        self.cache_size = cache_size
        self.finetune = finetune
        self.use_cache = use_cache
        self.data = h5py.File(self.hdf5_file, 'r')
        self.keys = list(self.data.keys())

        self.index_map = []
        for key in self.keys:
            group_size = len(self.data[key]['X'])  # Always assume 'X' is present
            self.index_map.extend([(key, i) for i in range(group_size)])
        
        # Cache to store recently accessed samples
        if self.use_cache:
            self.cache = {}
            self.cache_queue = deque(maxlen=self.cache_size)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        if self.use_cache and index in self.cache:
            cached_data = self.cache[index]
            if self.finetune:
                X, Y = cached_data
            else:
                X = cached_data
        else:
            group_key, sample_idx = self.index_map[index]
            grp = self.data[group_key]
            X = grp["X"][sample_idx]
            X = torch.FloatTensor(X)

            if self.finetune:
                Y = grp["y"][sample_idx]
                Y = torch.LongTensor([Y]).squeeze()
                if self.use_cache:
                    self.cache[index] = (X, Y)
            else:
                if self.use_cache:
                    self.cache[index] = X

            if self.use_cache:
                self.cache_queue.append(index)
        
        if self.squeeze:
            X = X.unsqueeze(0)
        
        if self.finetune:
            return X, Y
        else:
            return X

    def __del__(self):
        self.data.close()