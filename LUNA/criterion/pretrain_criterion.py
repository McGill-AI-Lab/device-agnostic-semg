"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PRETRAIN_CRITERION.PY - RECONSTRUCTION LOSS          ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Loss function for masked autoencoder pretraining. Computes reconstruction error
   separately for masked and unmasked regions of the input signal.

HIGH-LEVEL OVERVIEW:
   During pretraining, LUNA masks patches of EEG signals and learns to reconstruct them.
   This criterion measures how well the model reconstructs both the masked regions
   (the primary learning signal) and unmasked regions (for consistency).

KEY CLASSES:
   
   PretrainCriterion(nn.Module):
   - Configurable loss function (L1, L2, or Smooth L1)
   - Computes separate losses for masked vs unmasked patches
   - Returns (masked_loss, unmasked_loss) tuple
   
   Supported loss types:
   - 'l1': Mean Absolute Error (L1 loss)
   - 'l2': Mean Squared Error (L2 loss)  
   - 'smooth_l1': Smooth L1 / Huber loss (robust to outliers)

TECHNICAL DETAILS:
   The forward pass takes:
   - reconstructed: Model's output signal [B, C, T]
   - original: Ground truth signal [B, C, T]
   - mask: Boolean mask [B, C, T] where True = masked
   
   Returns:
   - masked_loss: Error only on masked regions
   - unmasked_loss: Error only on non-masked regions

USAGE IN TRAINING:
   Typically combined as: total_loss = masked_loss + λ * unmasked_loss
   where λ is a small coefficient (e.g., 0.1) to emphasize masked reconstruction.

RELATED FILES:
   - tasks/pretrain_task.py: Uses this criterion for pretraining
   - tasks/pretrain_task_LUNA.py: LUNA-specific pretraining implementation
   - criterion/query_specialization_criterion.py: Additional auxiliary loss
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
from torch import nn
import torch.nn.functional as F

class PretrainCriterion(nn.Module):
    """
    Criterion module to compute masked reconstruction losses.

    Args:
        loss_type (str): Type of loss to compute. Options are:
                         'l1' - L1 loss (Mean Absolute Error)
                         'l2' - L2 loss (Mean Squared Error)
                         'smooth_l1' - Smooth L1 loss (Huber loss)
    """
    def __init__(self, loss_type):
        super(PretrainCriterion, self).__init__()
        if loss_type not in ['l1', 'l2', 'smooth_l1']:
            raise ValueError("Invalid loss_type. Choose 'l1', 'l2', or 'smooth_l1'.")
        self.loss_type = loss_type

    def forward(self, reconstructed, original, mask):
        """
        Calculate loss between reconstructed and original signals,
        separately for masked and unmasked elements.

        Args:
            reconstructed (torch.Tensor): The reconstructed output from the model.
            original (torch.Tensor): The original input signal.
            mask (torch.BoolTensor): Boolean mask indicating which elements are masked.

        Returns:
            tuple: (masked_loss, unmasked_loss)
        """
        if self.loss_type == 'l1':
            # Mean Absolute Error on masked and unmasked elements
            masked_loss = F.l1_loss(reconstructed[mask], original[mask], reduction='mean')
            unmasked_loss = F.l1_loss(reconstructed[~mask], original[~mask], reduction='mean')
        elif self.loss_type == 'l2':
            # Mean Squared Error on masked and unmasked elements
            masked_loss = F.mse_loss(reconstructed[mask], original[mask], reduction='mean')
            unmasked_loss = F.mse_loss(reconstructed[~mask], original[~mask], reduction='mean')
        elif self.loss_type == 'smooth_l1':
            # Smooth L1 (Huber) loss on masked and unmasked elements
            masked_loss = F.smooth_l1_loss(reconstructed[mask], original[mask], reduction='mean')
            unmasked_loss = F.smooth_l1_loss(reconstructed[~mask], original[~mask], reduction='mean')

        return masked_loss, unmasked_loss
