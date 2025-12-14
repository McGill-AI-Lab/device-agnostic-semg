"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CHANNEL_EMBEDDINGS.PY - EEG CHANNEL NAMING & LOCATIONS               ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Central registry for EEG channel names across datasets, with utilities to map
   channel names to indices and retrieve 3D spatial coordinates for electrodes.

HIGH-LEVEL OVERVIEW:
   Different EEG datasets use different channel naming conventions (e.g., "FP1" vs "FP1-F7").
   This module:
   1. Defines channel lists for SEED, TUEG (bipolar), and Siena datasets
   2. Creates a unified mapping: channel_name → integer index
   3. Provides functions to get 3D (x, y, z) electrode locations
   4. Implements learnable channel embeddings for pretraining

KEY COMPONENTS:
   
   Channel Lists:
   - SEED_CHANNEL_LIST: 62 channels (monopolar, e.g., "FP1", "FP2")
   - TUEG_CHANNEL_LIST: 22 channels (bipolar, e.g., "FP1-F7", "T3-T5")
   - SIENA_CHANNEL_LIST: 30 channels (monopolar)
   
   Global Mappings:
   - CHANNEL_NAMES_TO_IDX: Unified dict mapping all channel names to indices
   - CHANNEL_IDX_TO_NAMES: Reverse mapping from indices to names

KEY FUNCTIONS:
   
   get_channel_indices(channel_names):
   - Input: List of channel name strings
   - Output: List of integer indices
   - Use: Convert dataset channels to model's internal indexing
   
   get_channel_locations(channel_names):
   - Input: List of channel names (monopolar or bipolar)
   - Output: List of (x, y, z) 3D coordinates
   - Uses MNE's standard_1005 montage for positions
   - For bipolar channels: averages positions of two electrodes
   
   ChannelEmbeddings(nn.Module):
   - Learnable embedding layer for channel identity
   - Used during pretraining to encode "which channel"
   - Input: channel indices (integers)
   - Output: embedding vectors [num_channels, embed_dim]

BIPOLAR MONTAGE HANDLING:
   Bipolar channels like "FP1-F7" represent the difference between two electrodes.
   For spatial location:
   - Parse "FP1-F7" → ["FP1", "F7"]
   - Get locations for FP1 and F7 from MNE montage
   - Average: location = (loc_FP1 + loc_F7) / 2
   
   This provides an approximate spatial position for bipolar derivations.

WHY CHANNEL LOCATIONS MATTER:
   - LUNA uses 3D positions as positional encodings
   - Helps model understand spatial relationships between electrodes
   - Enables topology-agnostic learning (model sees positions, not names)
   - Critical for generalizing across different montages

RELATED FILES:
   - models/LUNA.py: Uses channel locations in prepare_tokens()
   - datasets/seed_v_dataset.py: Computes channel locations for SEED-V
   - models/modules/channel_location_embedder.py: Processes locations into embeddings
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
from torcheeg.datasets.constants import SEED_CHANNEL_LIST
import mne 

SEED_PRETRAINING_CHANNEL_LIST = SEED_CHANNEL_LIST
TUEG_CHANNEL_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "A1-T3",
    "T4-A2",
]
SIENA_CHANNEL_LIST = [
    "FP1",
    "FP2",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FC1",
    "FC5",
    "CP1",
    "CP5",
    "F9",
    "FZ",
    "CZ",
    "PZ",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
    "FC2",
    "FC6",
    "CP2",
    "CP6",
    "F10",
]

all_channels = set()
for ds in [
    SEED_PRETRAINING_CHANNEL_LIST,
    TUEG_CHANNEL_LIST,
    SIENA_CHANNEL_LIST,
]:
    for ch in ds:
        all_channels.add(ch)
CHANNEL_NAMES_TO_IDX = {ch: i for i, ch in enumerate(sorted(all_channels))}
CHANNEL_IDX_TO_NAMES = {i: ch for ch, i in CHANNEL_NAMES_TO_IDX.items()}

def get_channel_indices(channel_names):
    indices = []
    for name in channel_names:
        indices.append(CHANNEL_NAMES_TO_IDX[name])
    return indices

def get_channel_names(channel_indices):
    names = []
    for idx in channel_indices:
        names.append(CHANNEL_IDX_TO_NAMES[idx])
    return names

def get_channel_locations(channel_names):
    if "-" in channel_names[0]:
        names = list(set([part for ch in channel_names for part in ch.split('-')]))
    else:
        names = channel_names
    ch_types = ['eeg'] * len(names)  # Channel types
    info = mne.create_info(ch_names=names, sfreq=256, ch_types=ch_types)
    info = info.set_montage(mne.channels.make_standard_montage("standard_1005"),match_case=False,match_alias={'cb1': 'POO7', 'cb2': 'POO8'})
    locs = []
    for name in channel_names:
        if name in TUEG_CHANNEL_LIST:
            electrode1, electrode2 = name.split('-')
            loc1 = info.get_montage().get_positions()['ch_pos'][electrode1]
            loc2 = info.get_montage().get_positions()['ch_pos'][electrode2]
            locs.append(((loc1 + loc2) / 2))
        else:
            locs.append(info.get_montage().get_positions()['ch_pos'][name])
    return locs

class ChannelEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(ChannelEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(len(CHANNEL_NAMES_TO_IDX), embed_dim)

    def forward(self, indices):
        return self.embeddings(indices)
    
    def initialize_weights(self):
        torch.init.normal_(self.embeddings.weight, std=2.0)

