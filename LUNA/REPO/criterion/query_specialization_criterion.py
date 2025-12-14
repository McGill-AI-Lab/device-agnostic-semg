"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 QUERY_SPECIALIZATION_CRITERION.PY - AUXILIARY LOSS           ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Auxiliary loss to encourage query specialization in LUNA's cross-attention mechanism.
   Promotes diversity by penalizing similarity between different query embeddings.

HIGH-LEVEL OVERVIEW:
   LUNA uses learnable queries in cross-attention to unify multi-channel EEG signals.
   This loss encourages each query to attend to different aspects of the input channels,
   preventing collapse where all queries learn the same representation.

KEY CLASSES:
   
   QuerySpecializationCriterion(nn.Module):
   - Computes pairwise similarity between query attention patterns
   - Penalizes off-diagonal elements (high similarity between different queries)
   - Encourages orthogonal/independent query specialization
   
   Parameters:
   - loss_type: 'l1', 'l2', or 'smooth_l1'
   - loss_coeff: Scaling factor for the loss (typically 0.1-1.0)

TECHNICAL DETAILS:
   Forward pass:
   1. Takes attention_scores [B, Q, C] where:
      - B = batch size
      - Q = number of queries (e.g., 4)
      - C = number of channels
   
   2. Computes query similarity matrix [B, Q, Q]:
      similarity = attention_scores @ attention_scores^T
   
   3. Masks diagonal (self-similarity) and penalizes off-diagonal:
      loss = mean(off_diagonal_elements)
   
   Goal: Minimize similarity between different queries → each query specializes

WHY THIS MATTERS:
   Without this loss, all queries might attend to the same channel patterns,
   reducing model capacity. With it, queries learn complementary representations,
   enabling richer unified embeddings.

RELATED FILES:
   - models/LUNA.py: CrossAttentionBlock produces attention_scores
   - tasks/pretrain_task_LUNA.py: Combines this with reconstruction loss
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

import torch
from torch import nn
import torch.nn.functional as F

class QuerySpecializationCriterion(nn.Module):
    def __init__(self, loss_type, loss_coeff=1.0):
        super(QuerySpecializationCriterion, self).__init__()
        if loss_type not in ['l1', 'l2', 'smooth_l1']:
            raise ValueError("Invalid loss_type. Choose 'l1', 'l2', or 'smooth_l1'.")
        self.loss_type = loss_type
        self.loss_coeff = loss_coeff

    def forward(self, attention_scores):
        B, Q, C = attention_scores.size() # B = batch size; Q = num queries; C = num channels
        query_similarity = torch.bmm(attention_scores, attention_scores.permute(0, 2, 1)) # (B, Q, Q)
        # Create a mask to zero out the diagonal elements
        mask = 1.0 - torch.eye(Q, device=attention_scores.device).unsqueeze(0) # Shape (1, Q, Q)
        # Zero out the diagonal elements of the similarity matrix
        off_diagonal_similarity = query_similarity * mask # mask broadcasts to (B, Q, Q)
        if self.loss_type == 'l1':
            loss = torch.mean(torch.abs(off_diagonal_similarity)) 
        elif self.loss_type == 'l2':
            loss = torch.mean(off_diagonal_similarity**2)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(off_diagonal_similarity, torch.zeros_like(off_diagonal_similarity))
        return loss * self.loss_coeff
