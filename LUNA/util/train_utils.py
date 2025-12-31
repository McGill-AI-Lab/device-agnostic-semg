"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TRAIN_UTILS.PY - TRAINING UTILITIES                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Utility functions for training: checkpoint management and robust data normalization.

KEY FUNCTIONS & CLASSES:
   
   1. find_last_checkpoint_path(checkpoint_dir):
      - Searches for "last.ckpt" in checkpoint directory
      - Used to resume training from last saved state
      - Returns None if no checkpoint found
      - Compatible with PyTorch Lightning checkpoint naming
   
   2. RobustQuartileNormalize:
      - Normalizes EEG signals using interquartile range (IQR)
      - Formula: (x - q_lower) / (q_upper - q_lower)
      - More robust to outliers than z-score normalization
      - Configured with quartile values (e.g., 0.025, 0.975)
      
      Why IQR normalization?
      - EEG often has artifacts (large spikes, movement)
      - Standard z-score sensitive to outliers
      - IQR focuses on central distribution, ignores extremes
      - Example: q_lower=0.025, q_upper=0.975 → use central 95% of data

USAGE:
   
   Resume Training:
   ```python
   checkpoint_path = find_last_checkpoint_path(cfg.checkpoint_dir)
   trainer.fit(model, datamodule, ckpt_path=checkpoint_path)
   ```
   
   Normalize Input:
   ```python
   normalizer = RobustQuartileNormalize(q_lower=0.025, q_upper=0.975)
   x_normalized = normalizer(x_raw)
   ```

RELATED FILES:
   - run_train.py: Uses find_last_checkpoint_path for resume
   - tasks/*: Use RobustQuartileNormalize for input preprocessing
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
#* Author:  Thorir Mar Ingolfsson                                             *
#* Author:  Anna Tegon                                                        *
#* Author:  Berkay Döner                                                      *
#*----------------------------------------------------------------------------*

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import Optional
import os
import os.path as osp
import torch
import torch.nn as nn


def find_last_checkpoint_path(checkpoint_dir: Optional[str]) -> Optional[str]:
    if checkpoint_dir is None:
        return None
    checkpoint_file_name = (
        f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}{ModelCheckpoint.FILE_EXTENSION}"
    )
    last_checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_file_name)
    if not osp.exists(last_checkpoint_filepath):
        return None

    return last_checkpoint_filepath

class RobustQuartileNormalize:
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper

    def __call__(self, tensor):
        iqr = self.q_upper - self.q_lower
        return (tensor - self.q_lower) / (iqr + 1e-8)