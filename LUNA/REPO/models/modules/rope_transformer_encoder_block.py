"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ROPE_TRANSFORMER_ENCODER_BLOCK.PY - TEMPORAL MODELING                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    
PURPOSE:
   Transformer encoder block with Rotary Position Embeddings (RoPE) for temporal
   sequence modeling. Core component of LUNA's temporal processing pipeline.

HIGH-LEVEL OVERVIEW:
   After channel unification via cross-attention, LUNA needs to model temporal
   dependencies across time patches. This module implements:
   1. Self-attention with RoPE (better than absolute position encoding)
   2. Feed-forward network for feature transformation
   3. Pre-normalization for training stability
   4. Residual connections and dropout for regularization

KEY CLASSES:
   
   1. RotarySelfAttentionBlock:
      - Multi-head self-attention with RoPE
      - Applies rotary embeddings to queries and keys
      - Benefits: Better length extrapolation, relative position encoding
   
   2. FeedForwardBlock:
      - Two-layer MLP with GELU activation
      - LayerNorm between layers for stability
      - Dropout for regularization
   
   3. RotaryTransformerBlock:
      - Complete encoder block: Attention → FFN
      - Pre-norm architecture (norm before attention/FFN)
      - DropPath for stochastic depth (training efficiency)

ROTARY POSITION EMBEDDINGS (RoPE):
   
   Why RoPE instead of absolute position encoding?
   ✓ Encodes relative positions naturally (attention sees distance between tokens)
   ✓ Better extrapolation to longer sequences than seen during training
   ✓ No learned position embeddings needed (reduces parameters)
   ✓ Works by rotating Q and K vectors in complex space
   
   Implementation:
   - Applies learned frequency-based rotations to each attention head
   - Rotation amount depends on token position
   - Result: attention weights naturally encode relative distances

ARCHITECTURE DETAILS:
   
   Input: [B, num_patches, embed_dim]
   
   Block Structure:
   x = x + DropPath(Attention(LayerNorm(x)))
   x = x + DropPath(FFN(LayerNorm(x)))
   
   Output: [B, num_patches, embed_dim]
   
   - Pre-norm: Normalizes before each sub-layer (more stable than post-norm)
   - Residual connections: Enables gradient flow in deep networks
   - DropPath: Randomly drops entire residual paths during training

WHY THIS MATTERS FOR EEG:
   EEG signals have rich temporal dynamics:
   - Brain states evolve over seconds
   - Transient events (e.g., epileptic spikes) have specific temporal patterns
   - Long-range dependencies matter (e.g., sleep spindles lasting 0.5-2s)
   
   RoPE-based transformers excel at:
   ✓ Capturing dependencies at multiple timescales
   ✓ Attending to relevant past context
   ✓ Generalizing to different sequence lengths

RELATED FILES:
   - models/LUNA.py: Stacks multiple RotaryTransformerBlocks
   - External: rotary_embedding_torch library (RoPE implementation)
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

from rotary_embedding_torch import RotaryEmbedding
import torch
import torch.nn as nn
from einops import rearrange
import torch
from timm.layers import DropPath


class RotarySelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim, learned_freq=False)

        self.scale = qk_scale or head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.attn_drop_fn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (K, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        # Calculate attention scores
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # Apply softmax to get attention probabilities
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply dropout
        attn_weights = self.attn_drop_fn(attn_weights)
        
        # Apply attention weights to values
        attn = attn_weights @ v  # (B, H, N, D)
        attn = rearrange(attn, "B H N D -> B N (H D)")
        return self.proj_drop(self.proj(attn))


class GEGLU(nn.Module):
    def __init__(self):
        super(GEGLU, self).__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class RotaryTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForwardBlock(
            dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=drop
        )

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
