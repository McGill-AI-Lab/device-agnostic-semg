# 🧠 LUNA CODEBASE OVERVIEW

This document provides a high-level map of the codebase to help you navigate and understand how all the pieces of the repository fit together.

---

## 📚 Quick Navigation

- [What is LUNA?](#what-is-luna)
- [Repository Structure](#repository-structure)
- [Key Concepts](#key-concepts)
- [Typical Workflows](#typical-workflows)
- [File-by-File Guide](#file-by-file-guide)

---

## What is LUNA?

**LUNA** (Linear-in-channels, Unified, Network-Agnostic) is a transformer-based foundation model for EEG analysis with two key innovations:

1. **Topology-Agnostic**: Works with any EEG electrode configuration (different channel counts, different montages)
2. **Linear Complexity**: Scales linearly with number of channels (not quadratically like standard transformers)

### How LUNA Works

```
Input: Multi-channel EEG [Batch, Channels, Time]
                ↓
        Patch Embedding (time patches)
                ↓
        Frequency Features (FFT-based)
                ↓
        Channel Location Encoding (3D positions)
                ↓
    ╔═════════════════════════════════╗
    ║  CROSS-ATTENTION (Key Step!)    ║
    ║  Q queries attend to C channels ║
    ║  Result: [B, patches, Q×D]      ║
    ╚═════════════════════════════════╝
                ↓
        Temporal Transformers (RoPE)
                ↓
    ┌─────────────────────────────┐
    │ Pretraining: Reconstruction │
    │ Finetuning: Classification  │
    └─────────────────────────────┘
```

---

## Repository Structure

```
LUNA/REPO/
├── run_train.py              ⭐ Main entry point - run this to train!
│
├── config/                   🔧 Hydra configurations
│   ├── defaults.yaml         └─ Base config
│   ├── experiment/           └─ Experiment-specific configs (pretrain/finetune)
│   ├── model/                └─ Model architectures (LUNA_base, large, huge)
│   ├── data_module/          └─ Data loading configurations
│   ├── task/                 └─ Training task configs
│   └── ...
│
├── models/                   🧠 Model architectures
│   ├── LUNA.py               ⭐ Main LUNA model implementation
│   └── modules/
│       ├── channel_embeddings.py        └─ Channel naming & 3D locations
│       ├── frequency_embedder.py        └─ FFT-based frequency features
│       ├── rope_transformer_encoder_block.py  └─ Temporal modeling
│       └── channel_location_embedder.py └─ Spatial position encoding
│
├── tasks/                    🎯 Training logic (PyTorch Lightning modules)
│   ├── pretrain_task_LUNA.py  ⭐ Masked autoencoder pretraining
│   ├── finetune_task_LUNA.py  ⭐ Classification finetuning
│   ├── pretrain_task.py        └─ Generic pretraining task
│   └── finetune_task.py        └─ Generic finetuning task
│
├── data_module/              📊 Data loading (PyTorch Lightning DataModules)
│   ├── finetune_data_module.py          └─ Simple train/val/test splits
│   ├── pretrain_data_module.py          └─ Multi-dataset pretraining
│   ├── subject_independent_data_module.py └─ Cross-subject evaluation
│   └── multiloader_data_module.py       └─ Variable channel counts
│
├── datasets/                 💾 Dataset implementations
│   ├── hdf5_dataset.py       ⭐ Efficient HDF5 loader (TUH datasets)
│   └── seed_v_dataset.py     └─ SEED-V emotion recognition
│
├── criterion/                📉 Loss functions
│   ├── pretrain_criterion.py           └─ Reconstruction loss
│   └── query_specialization_criterion.py └─ Query diversity loss
│
├── schedulers/               📈 Learning rate schedules
│   ├── cosine.py             └─ Cosine annealing (recommended)
│   └── multi_step_lr.py      └─ Step decay
│
├── util/                     🛠️ Utilities
│   └── train_utils.py        └─ Checkpoint management, normalization
│
└── make_datasets/            🔄 Data preprocessing
    ├── process_raw_eeg.py    └─ EDF → pickle (windowing, bipolar montage)
    └── make_hdf5.py          └─ Pickle → HDF5 (efficient storage)
```

---

## Key Concepts

### 1. Cross-Attention Channel Unification

**Problem**: Traditional EEG models require fixed channel configurations.

**LUNA's Solution**: Use Q learnable queries (e.g., 4) to attend to C channels:
- Complexity: O(Q × C) per patch (linear in C!)
- Each query learns a different aspect of channel patterns
- Result: Unified representation independent of channel count

### 2. Patch-Based Processing

Like Vision Transformers (ViT), LUNA splits EEG signals into patches:
- Time dimension: e.g., 20s signal → 128 patches of 40 samples each
- Each patch processed independently initially
- Temporal dependencies captured by transformer afterward

### 3. Masked Autoencoder Pretraining

Self-supervised learning on unlabeled EEG:
1. Mask random patches (e.g., 50%)
2. Model reconstructs masked content
3. Learns rich representations of EEG structure
4. Transfer to downstream tasks

### 4. Multi-Stage Training

```
Stage 1: Pretraining (weeks on GPUs)
└─ Large unlabeled datasets (TUEG + Siena, 21k+ hours)
└─ Masked reconstruction objective
└─ Saves: pretrained encoder weights

Stage 2: Finetuning (hours on GPUs)
└─ Smaller labeled datasets (TUAB/TUAR/TUSL)
└─ Classification objective
└─ Load pretrained weights, add classification head
```

---

## Typical Workflows

### Workflow 1: Training LUNA from Scratch

```bash
# 1. Prepare datasets
python make_datasets/process_raw_eeg.py tueg --root_dir /data/TUEG/edf --output_dir /processed
python make_datasets/make_hdf5.py --prepath /processed --dataset All

# 2. Pretrain (takes days/weeks!)
python run_train.py +experiment=LUNA_pretrain /model=LUNA_base

# 3. Finetune on TUAB
python run_train.py +experiment=LUNA_finetune /model=LUNA_base \
    pretrained_safetensors_path=/path/to/LUNA_base.safetensors
```

### Workflow 2: Using Pretrained Weights

```bash
# 1. Download weights from Hugging Face
from huggingface_hub import snapshot_download
snapshot_download(repo_id="thorir/LUNA", local_dir="checkpoints/LUNA")

# 2. Prepare your dataset (TUAB example)
python make_datasets/process_raw_eeg.py tuab --root_dir /data/TUAB/edf --output_dir /processed
python make_datasets/make_hdf5.py --prepath /processed --dataset TUAB

# 3. Finetune on your task
python run_train.py +experiment=LUNA_finetune /model=LUNA_base \
    pretrained_safetensors_path=checkpoints/LUNA/Base/LUNA_base.safetensors \
    data_module.train.hdf5_file=/processed/TUAB_data/train.h5 \
    data_module.val.hdf5_file=/processed/TUAB_data/val.h5
```

### Workflow 3: Adapting to New Dataset

```python
# 1. Create custom dataset (if needed)
# See datasets/seed_v_dataset.py as example

# 2. Update config
# config/experiment/my_experiment.yaml:
"""
defaults:
  - override /data_module: subject_independent_data_module
  - override /model: LUNA_base

model:
  num_classes: 5  # Your number of classes

data_module:
  datasets:
    my_dataset:
      _target_: datasets.my_dataset.MyDataset
      root_path: /path/to/data
"""

# 3. Run finetuning
python run_train.py +experiment=my_experiment \
    pretrained_safetensors_path=checkpoints/LUNA/Base/LUNA_base.safetensors
```

---

## File-by-File Guide

### 🚀 Start Here

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_train.py` | Main training script | Every training run |
| `models/LUNA.py` | Core model architecture | Understanding model structure |
| `README.md` | Project overview | First-time setup |

### 🔧 Configuration (config/)

| Directory | Purpose |
|-----------|---------|
| `experiment/` | Complete experiment configs (pretrain/finetune) |
| `model/` | Model size variants (base/large/huge) |
| `data_module/` | Data loading configurations |
| `task/` | Training task settings |
| `criterion/` | Loss function configs |
| `scheduler/` | Learning rate schedule configs |

**Key file**: `defaults.yaml` - Base configuration that others override

### 🧠 Models (models/)

| File | Description | Key Classes |
|------|-------------|-------------|
| `LUNA.py` | Main architecture | `LUNA`, `CrossAttentionBlock`, `PatchEmbedNetwork` |
| `modules/channel_embeddings.py` | Channel naming & locations | `get_channel_locations()`, `ChannelEmbeddings` |
| `modules/frequency_embedder.py` | FFT features | `FrequencyFeatureEmbedder` |
| `modules/rope_transformer_encoder_block.py` | Temporal modeling | `RotaryTransformerBlock` |
| `modules/channel_location_embedder.py` | Spatial encoding | `ChannelLocationEmbedder` |

**Read First**: `LUNA.py` for overall architecture

### 🎯 Tasks (tasks/)

| File | Purpose | When to Use |
|------|---------|-------------|
| `pretrain_task_LUNA.py` | MAE pretraining | Training from scratch |
| `finetune_task_LUNA.py` | Classification | Using pretrained weights |
| `pretrain_task.py` | Generic pretraining | Custom pretraining objectives |
| `finetune_task.py` | Generic finetuning | Custom finetuning |

**Architecture**: PyTorch Lightning `LightningModule` that wraps model + training logic

### 📊 Data Loading (data_module/ & datasets/)

#### Data Modules (high-level data management)

| File | Use Case |
|------|----------|
| `finetune_data_module.py` | Pre-split train/val/test (TUAB/TUAR/TUSL) |
| `pretrain_data_module.py` | Multi-dataset pretraining (TUEG+Siena) |
| `subject_independent_data_module.py` | Cross-subject evaluation (SEED-V) |
| `multiloader_data_module.py` | Varying channel counts |

#### Datasets (low-level data loading)

| File | Use Case |
|------|----------|
| `hdf5_dataset.py` | TUH datasets (TUAB/TUAR/TUSL) |
| `seed_v_dataset.py` | SEED-V emotion recognition |

### 📉 Loss Functions (criterion/)

| File | Loss Type | Used For |
|------|-----------|----------|
| `pretrain_criterion.py` | Reconstruction | Pretraining |
| `query_specialization_criterion.py` | Auxiliary | Query diversity during pretraining |

### 📈 Schedulers (schedulers/)

| File | Type | Recommended For |
|------|------|-----------------|
| `cosine.py` | Cosine annealing | Most use cases (smooth decay) |
| `multi_step_lr.py` | Step decay | Baselines, reproducibility |

### 🔄 Data Preprocessing (make_datasets/)

| File | Input | Output | Step |
|------|-------|--------|------|
| `process_raw_eeg.py` | EDF files | Pickle files | 1st |
| `make_hdf5.py` | Pickle files | HDF5 files | 2nd |

**Pipeline**: Raw EDF → Pickles (windowed) → HDF5 (efficient storage)

---

## Understanding the Training Flow

### Pretraining Flow

```
1. run_train.py
   └─ Loads config from config/experiment/LUNA_pretrain.yaml
   └─ Instantiates PretrainDataModule (multi-dataset)
   └─ Instantiates pretrain_task_LUNA.MaskTask
       └─ Contains LUNA model (num_classes=0)
       └─ Contains PretrainCriterion
       └─ Contains QuerySpecializationCriterion
   
2. Training Loop (PyTorch Lightning)
   └─ For each batch:
       a) Generate random mask
       b) Forward: model reconstructs signal
       c) Compute loss (masked + unmasked + query_spec)
       d) Backward & update weights
   
3. Validation Loop
   └─ Same as training but no updates
   └─ Log visualizations (original vs reconstructed)

4. Save Checkpoint
   └─ Best model based on validation loss
   └─ Can be loaded for finetuning
```

### Finetuning Flow

```
1. run_train.py
   └─ Loads config from config/experiment/LUNA_finetune.yaml
   └─ Instantiates FinetuneDataModule (labeled data)
   └─ Instantiates finetune_task_LUNA.FinetuneTask
       └─ Contains LUNA model (num_classes>0)
       └─ Loads pretrained encoder weights
       └─ Freezes encoder (optional) or finetunes all
   
2. Training Loop
   └─ For each batch:
       a) Forward: model predicts class
       b) Compute cross-entropy loss
       c) Update weights (head + optionally encoder)
       d) Log metrics (accuracy, AUROC, etc.)
   
3. Validation/Test
   └─ Compute metrics on held-out data
   └─ Return AUROC, AUPR, F1, etc.
```

---

## Configuration System (Hydra)

LUNA uses Hydra for hierarchical configuration:

```yaml
# config/experiment/LUNA_finetune.yaml
defaults:
  - override /data_module: finetune_data_module
  - override /model: LUNA_base
  - override /task: finetune_task_LUNA

# Can override any parameter:
model:
  num_classes: 2  # TUAB binary classification

data_module:
  train:
    hdf5_file: ${env:DATA_PATH}/TUAB_data/train.h5
```

**Override from command line**:
```bash
python run_train.py +experiment=LUNA_finetune /model=LUNA_large model.num_classes=4
```

---

## Common Questions

### Q: How do I add a new dataset?

**A**: Three steps:

1. Create dataset class in `datasets/`:
```python
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {
            'input': signal,  # [C, T]
            'label': label,
            'channel_locations': locations  # [C, 3]
        }
```

2. Create config in `config/dataset/my_dataset.yaml`

3. Use in experiment config or command line

### Q: Which files should I modify for my research?

**Most common modifications**:
- New loss function → Add to `criterion/`
- New architecture module → Add to `models/modules/`
- New task/evaluation → Add to `tasks/`
- New dataset → Add to `datasets/` and `data_module/`

**Don't modify** (unless necessary):
- `run_train.py` (stable entry point)
- Core LUNA architecture (unless experimenting with model changes)

### Q: How do I debug issues?

1. **Config issues**: Run with `--cfg job` to print full config
2. **Data issues**: Check `data_module` logs, inspect HDF5 files
3. **Model issues**: Add print statements in `models/LUNA.py`
4. **Training issues**: Check TensorBoard logs, adjust learning rate

### Q: How do I visualize training?

```bash
# Start TensorBoard
tensorboard --logdir outputs/

# Logs include:
# - Training/validation loss curves
# - Learning rate schedule
# - Metric curves (AUROC, accuracy, etc.)
# - Signal visualizations (pretraining only)
```

---

## Tips for New Users

1. **Start with pretrained weights**: Don't pretrain from scratch unless you have massive compute

2. **Use small model first**: `LUNA_base` is fast for debugging, upgrade to `large`/`huge` later

3. **Check data first**: Always verify your HDF5 files load correctly before training

4. **Monitor validation loss**: If it diverges from training loss, you may have data leakage or overfitting

5. **Use subject-independent splits**: For fair evaluation on EEG (subjects are very different!)

6. **Normalize your inputs**: EEG has high variance across subjects, always normalize

7. **Start with cosine scheduler**: Works well in most cases, easier than tuning step milestones

---

## Getting Help

- **Paper**: [LUNA: Efficient and Topology-Agnostic Foundation Model for EEG](https://arxiv.org/abs/2510.22257)
- **Pretrained Weights**: [Hugging Face Model Hub](https://huggingface.co/thorir/LUNA)
- **Issues**: [GitHub Issues](https://github.com/pulp-bio/BioFoundation/issues)

---

## Summary

This codebase implements LUNA, a topology-agnostic EEG foundation model using:
- **Cross-attention** for channel unification (linear complexity)
- **Masked autoencoder** pretraining (self-supervised)
- **Flexible architecture** supporting any EEG montage
- **PyTorch Lightning** + **Hydra** for clean, modular code

**Key innovation**: Query-based channel unification allows a single model to work across
different EEG datasets with different channel configurations, eliminating the need for
dataset-specific architectures.

Happy coding! 🧠⚡

