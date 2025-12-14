"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         FREQUENCY_EMBEDDER.PY - FREQUENCY-DOMAIN PATCH FEATURES              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Augments patch embeddings with frequency-domain information using FFT.
   Captures spectral characteristics (power, phase) that are important for EEG analysis.

HIGH-LEVEL OVERVIEW:
   EEG signals contain critical information in frequency bands (delta, theta, alpha, etc.).
   This module:
   1. Splits input signal into patches (like LUNA's patch embedding)
   2. Applies FFT to each patch
   3. Extracts magnitude (power) and phase
   4. Projects frequency features to embedding dimension
   5. Adds these to existing patch embeddings

KEY CLASS:
   
   FrequencyFeatureEmbedder(nn.Module):
   - Input: [B, C, T] raw EEG signal
   - Process:
     a) Reshape into patches [B, C, S, P] where P=patch_size
     b) Apply rfft (real FFT) to each patch
     c) Extract magnitude = |FFT| and phase = angle(FFT)
     d) Concatenate magnitude and phase
     e) Project to embed_dim using MLP
   - Output: [B, C×S, embed_dim] frequency embeddings

TECHNICAL DETAILS:
   
   FFT Representation:
   - rfft: Real FFT for real-valued signals (more efficient than full FFT)
   - Output size: patch_size//2 + 1 frequency bins
   - Magnitude: Captures power at each frequency
   - Phase: Captures timing information
   
   Feature Size:
   - Magnitude bins: patch_size//2 + 1
   - Phase bins: patch_size//2 + 1
   - Total features: 2 * (patch_size//2 + 1)
   
   MLP Projection:
   - Input dim: 2 * (patch_size//2 + 1)
   - Hidden dim: 4× input dim
   - Output dim: embed_dim
   - Activation: GELU (smooth, non-saturating)

WHY FREQUENCY FEATURES MATTER FOR EEG:
   EEG brain states are characterized by frequency bands:
   - Delta (0.5-4 Hz): Deep sleep
   - Theta (4-8 Hz): Drowsiness, meditation
   - Alpha (8-13 Hz): Relaxed wakefulness
   - Beta (13-30 Hz): Active thinking
   - Gamma (30+ Hz): Cognitive processing
   
   By adding frequency features, LUNA can:
   ✓ Better distinguish between brain states
   ✓ Detect oscillatory patterns
   ✓ Capture both temporal and spectral information

INTEGRATION WITH LUNA:
   In LUNA.prepare_tokens():
   1. Patch embedding: x_patched = patch_embed(x_signal)
   2. Frequency embedding: freq_embed = freq_embed(x_signal)
   3. Combined: x_patched = x_patched + freq_embed
   
   This creates richer patch representations that encode both time-domain
   waveform shape and frequency-domain power spectrum.

RELATED FILES:
   - models/LUNA.py: Combines patch + frequency embeddings
   - models/modules/rope_transformer_encoder_block.py: Processes combined embeddings
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
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
from timm.layers import Mlp

class FrequencyFeatureEmbedder(nn.Module):
    """
    This class takes data that is of the form (B, C, T) and patches it 
    along the time dimension (T) into patches of size P (patch_size).
    The output is of the form (B, C, S, P) where S = T // P.
    """
    def __init__(self, patch_size, embed_dim):
        super(FrequencyFeatureEmbedder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        in_features = 2*(patch_size // 2 + 1)
        self.frequency_to_embed = Mlp(in_features=in_features, hidden_features=int(4*in_features), out_features=embed_dim)

    def forward(self, x):
        B, C, T = x.size()
        S = T // self.patch_size
        # There is a chance that the input tensor is not divisible by the patch size
        # In this case we need to pad the tensor with zeros
        if T % self.patch_size != 0:
            # Pad last dimension with zeros to make it divisible by patch size
            pad_size = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, pad_size))
            T = x.size(-1)
            S = T // self.patch_size   
        x = x.view(B, C, S, self.patch_size)

        freq_representation = fft.rfft(x, dim=-1)  # (B, C, num_patches, patch_size // 2 + 1)
        magnitude = torch.abs(freq_representation)
        phase = torch.angle(freq_representation)    
        
        # Concatenate magnitude and phase along the frequency axis (last dimension)
        freq_features = torch.cat((magnitude, phase), dim=-1)
        # Map frequency features to embedding dimension
        embedded = self.frequency_to_embed(freq_features)  # (B, C, num_patches, embed_dim)
        embedded = rearrange(embedded, 'B C t D -> B (C t) D')
        return embedded
