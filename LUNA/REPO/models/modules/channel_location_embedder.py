"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CHANNEL_LOCATION_EMBEDDER.PY - SPATIAL POSITION ENCODING             ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Embeds 3D spatial coordinates of EEG electrodes into the model's feature space.
   Provides topology-aware positional information for channel-agnostic learning.

HIGH-LEVEL OVERVIEW:
   EEG electrodes are positioned at specific locations on the scalp (x, y, z coordinates).
   This module:
   1. Takes raw 3D positions [B, C, 3]
   2. Projects them through a 2-layer MLP
   3. Outputs embeddings [B, C, embed_dim]
   4. These are added to patch embeddings to encode "where" each channel is

KEY CLASS:
   
   ChannelLocationEmbedder(nn.Module):
   - Input: channel_locations [B, num_channels, 3]
     * 3D = (x, y, z) coordinates in standard head coordinate system
   - Architecture:
     * Linear: 3 → embed_dim
     * LayerNorm
     * Linear: embed_dim → embed_dim
     * LayerNorm
   - Output: [B, num_channels, embed_dim]

WHY SPATIAL EMBEDDINGS MATTER:
   
   EEG patterns have spatial structure:
   - Motor activity: localized to motor cortex (central electrodes)
   - Visual processing: occipital lobe (posterior electrodes)
   - Language: left temporal lobe
   - Epileptic foci: specific spatial origins
   
   By encoding positions, LUNA can:
   ✓ Learn spatial priors (e.g., frontal vs. posterior patterns)
   ✓ Generalize across montages (different layouts, same spatial relationships)
   ✓ Attend to spatially relevant channels

INTEGRATION WITH LUNA:
   In LUNA.prepare_tokens():
   1. Get raw locations from dataset
   2. Normalize to [0, 1] range: (loc - min) / (max - min)
   3. Add noise during pretraining: loc += randn() * 0.02
   4. Apply NeRF-style positional encoding
   5. Project with this embedder
   6. Add to patch embeddings
   
   Result: Each channel token knows both "what" (signal content) and "where" (position).

COMPARISON TO OTHER APPROACHES:
   
   Fixed Channel Models:
   - Hard-code channel order (e.g., "FP1 is always index 0")
   - Can't handle missing channels or different montages
   
   LUNA with Location Embeddings:
   - Soft spatial encoding (positions as continuous features)
   - Works with any subset of channels
   - Learns to pool information spatially
   
   This is similar to positional encoding in NLP transformers, but for spatial
   dimensions instead of sequence positions.

RELATED FILES:
   - models/LUNA.py: Uses this in prepare_tokens() with NeRF encoding
   - models/modules/channel_embeddings.py: Gets raw 3D coordinates
   - datasets/*: Provide channel_locations in batch dict
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

class ChannelLocationEmbedder(nn.Module):
    def __init__(
        self,
        channel_locations_dim: int = 3,
        in_chans: int = 22,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.channel_locations_dim = channel_locations_dim
        self.embed_dim = embed_dim
        self.channel_embeddings = nn.ModuleList([
                nn.Linear(channel_locations_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            ])
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, channel_locations):
        """
        Output shape:
            [in_chans, embed_dim]
        Args:
            channel_locations: [B, in_chans, channel_locations_dim]
        """
        
        out = channel_locations
        for layer in self.channel_embeddings:
            out = layer(out)
        return out
