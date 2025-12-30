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

## LUNA Breakdown

**A complete ground-up explanation of how LUNA works**

### Part 1 — Background you need

#### 1. What are EEG and EMG?

**EEG (electroencephalography)**
- Electrodes on the **scalp**.
- Measures tiny voltage changes coming from brain activity over time.
- You get a **signal for each electrode**: "channel 1's voltage over time, channel 2's voltage over time, …".

**EMG (electromyography)**
- Electrodes on **muscles** (skin surface or needles).
- Measures electrical activity when muscles contract/relax.
- Same idea: **multiple channels**, each is a time series.

In code, both can be thought of as:
- a **matrix**: `channels × time`
- e.g. 32 electrodes, 10,000 time points → shape `(32, 10000)`

So *mathematically*, EEG and sEMG data are very similar: **multi-channel time series signals**.

---

#### 2. Why is channel layout a problem?

Imagine 3 hospitals/datasets:
- Hospital A: 19 EEG electrodes
- Hospital B: 21 electrodes, slightly different positions
- Hospital C: 32 electrodes, different positions again

You want **one model** that works on all three. Problems:

1. The **number of channels is different** (19 vs 21 vs 32).
2. The **names and positions** differ (Fp1, Fp2, Cz, etc.).
3. Traditional neural nets like to assume:
   - fixed input size
   - fixed order (channel 1, 2, 3… always mean the same thing)

This is what the LUNA paper calls **"topological heterogeneity"**: different "wiring diagrams" / layouts of electrodes across datasets.

For sEMG it's even messier:
- Different devices: 8 channels, 16 channels, 64-channel electrode grids…
- Often you **don't even know exact 3D coordinates** of each electrode.

---

#### 3. Mini deep learning refresher (only what you need here)

**3.1 Neural network, embeddings, latent space**

- A neural network: series of layers that take input → transform → output.
- An **embedding** is just: "take some raw data and map it to a **vector of numbers** that hopefully captures useful structure."
- A **latent representation** (or latent space): the internal vectors the model learns; they're not directly human-interpretable, but they summarize what the model has understood so far.

In our context:
- Raw: `channels × time` signals
- Latent: for each time chunk, a "summary vector" that describes what's happening across channels.

**3.2 Attention and transformers (very high-level)**

Classic "Attention is All You Need" attention:
- You have **queries**, **keys**, and **values**:
  - Query = "what am I looking for?"
  - Key = "what do I contain?"
  - Value = "what information should I give if you focus on me?"
- Attention computes **how strongly each query should look at each key**, then mixes the values accordingly.

In simple words: **a smart weighted average** of information, where the weights depend on similarity between query and key.

**Self-attention**: queries/keys/values all come from the **same sequence** (e.g. words in a sentence, or time steps in a signal).

**Cross-attention**: queries come from one place, keys/values from another (e.g. learned queries attending to channels).

Transformers are just stacks of layers that use attention plus small MLPs.

---

#### 4. What does "self-supervised masked reconstruction" mean?

LUNA is **pretrained** without labels by playing a "mask and reconstruct" game:

1. Take the signal.
2. **Mask out** (hide) some parts.
3. Ask the model to **reconstruct the missing parts** at the output.
4. Loss = difference between true signal vs reconstructed signal on the masked region.

Why this is useful:
- Model is forced to learn **structure in the data** (how channels relate, temporal patterns) to fill in the gaps.
- Then you can fine-tune on small labeled datasets later.

---

### Part 2 — How LUNA works, step by step

Now let's walk through LUNA as if it's processing **one EEG recording**. sEMG can plug into the same pipeline with changes later.

Think of the input as:

> **A spreadsheet with C rows (channels) and T columns (time samples).**

So the input shape is: **(Batch, Channels, Time)** = `(B, C, T)`.

---

#### Step 0 — Split time into "patches"

Instead of looking at all T time points at once, LUNA divides the time axis into **chunks**:

- Example: patch size = 40 samples
- Then T = 4000 samples → 4000 / 40 = 100 patches.

So each channel is cut into small windows:

```
Channel 1: patch1, patch2, … patch100
Channel 2: patch1, patch2, … patch100
…
```

This is similar to ViT (Vision Transformer) where an image is split into patches, but here we split **time**.

---

#### Step 1 — Extract features from each patch (per channel)

For each patch, LUNA builds a **feature vector** that captures:

**1. Time-domain pattern**: using a small 1D convolution network
- Think: it slides small filters across the patch and detects patterns (like edges in images, but here "bursts", "spikes", etc. in the signal).

**2. Frequency-domain pattern**: using the FFT
- Take the patch → compute its frequency content (how much of each frequency is present).
- Get magnitude and phase and feed them through a little MLP to create another vector.

Then it **adds time features + frequency features** together → this is the **patch embedding**.

So now for every **channel** and **time patch**, we have a vector like:

> "What this channel is doing during this time window, in terms of shape and rhythm."

At this moment we have something like:
- shape ~ `(B, C, S, D)`
  - S = number of patches in time
  - D = embedding dimension

This is still **channel-specific** and **not invariant to how many channels there are**.

---

#### Step 2 — (Optional) encode where each channel is on the head

For EEG, each electrode has **3D coordinates** (x,y,z on a sphere). LUNA:

1. Takes these coordinates.
2. Feeds them through a sinusoidal "NeRF-style" encoder + MLP.
3. Adds that to the patch embeddings.

So each patch embedding now knows not only "what the signal looks like", but also "where on the head this channel is".

For **sEMG**, often we **don't know** these positions, so you might:
- Use dummy/zero coordinates, or
- Replace this with a learned channel ID embedding.

But conceptually: this step is "attach spatial information".

---

#### Step 3 — The big idea: **unify variable channels into a fixed-size set of "queries"**

This is the core LUNA trick.

**3.1 Why do we need this?**

If we feed all `(C × S)` tokens into a normal transformer, attention scales with **(C × S)²**:
- With many channels and many patches, that's huge.
- Also, different recordings have different C, which complicates things.

LUNA wants:
- **A fixed small number Q of "slots" per time patch**, regardless of how many channels C there are.
- So we can run the main transformer only on these Q slots → cheaper and layout-agnostic.

**3.2 How do learned queries work?**

For each time patch (say patch #17):
- You have `C` patch embeddings (one per channel) → think of them as **C "sources"** of information.
- LUNA also has **Q learnable "query vectors"** (e.g., Q = 4 or 8) that are shared across all recordings.

**Analogy:**
- Each **channel embedding** is like a **sensor** giving a report.
- Each **query** is like an **expert journalist** with a particular "interest":
  - Query 1 might learn to look for "frontal region, low-frequency activity".
  - Query 2 might learn to look for "broad overview across all channels".
  - etc.

They use **cross-attention**:
- Queries = the "journalists"
- Keys/values = the "sensor embeddings" (channels)

Each query:
1. Computes attention weights over all C channels ("which channels matter to me?").
2. Mixes their values accordingly ("what do I conclude from them?").

**Result for that patch:**
- We started with **C variable channels**.
- We end with **Q fixed query outputs** (one vector per query).

So per patch we now have **Q vectors** that summarize all channels.
This gives shape ~ `(B, S, Q, D)`.

**Important**: **Q is fixed**, even if C changes → that's how we get **channel-count invariance**.

Also, because cross-attention treats channels as a **set** (not caring about order), the model doesn't rely on the exact channel ordering.

---

#### Step 4 — Temporal transformer over patches

Now we want to model **how things change over time**.

At this point, for each time patch you have **Q vectors** that summarize all channels.

LUNA then:

1. Flattens or rearranges the Q vectors per patch into a single representation per patch (or keeps them as multiple slots; implementation varies slightly but conceptually it's "Q slots per patch").
2. Stacks patches in time: patch1, patch2, … patchS.
3. Runs a **Transformer over the sequence of patches**, using **temporal positional encodings** (RoPE) to keep track of order.

So now the model can learn long-range temporal patterns like:
- "Pattern A happens in patches 10–15, then pattern B appears later."
- "Slow drift across seconds/minutes."

Crucially: this attention is over **S patches only**, not `(C×S)` tokens.
So complexity is more like **S² + C·S** (roughly) instead of `(C·S)²`.
**Big win when C is large.**

---

#### Step 5 — Decoders (pretraining vs downstream)

**5.1 Pretraining: reconstruct masked patches**

For self-supervised pretraining, the task is: **reconstruct the masked portions** of the input signal.

1. At input, they randomly mask some patches (replace with a learned "mask token" or something similar).
2. After the transformer, they have latent representations for each patch.

To reconstruct **per-channel** signals, they:
- Create **per-channel decoder queries** (something like a learned embedding for each channel).
- Cross-attend these decoder queries to the latent.
- Produce an output that has the same shape as the original signal (`C × T`).

Then compute a reconstruction loss on the masked parts:
- "How close is the reconstructed signal to the true original signal for those masked patches?"

This trains the whole system (patch embeddings, queries, transformer) to understand structure.

**5.2 Downstream tasks: classification or other labels**

Once pretrained, they can swap the decoder head:
- Instead of reconstructing the signal, they:
  - Pool or attend over the latent representations.
  - Add a simple classification layer on top.
  - Fine-tune on tasks like "normal vs abnormal EEG", "artifact vs clean", etc.

For your sEMG project, you'd change this head to do things like:
- pose regression (fingers/hand),
- gesture classification,
- maybe even continuous control signals.

---

#### Step 6 — Extra regularization: "query specialization"

They add a small penalty so that:
- Each of the Q queries doesn't become redundant.
- Encourages the queries to learn **different roles**, e.g.
  - one query focusing more on frontal electrodes,
  - another on temporal,
  - another broad, etc.

Technically: they look at attention patterns of queries and discourage them from being too similar to each other.

Conceptually: "make the journalists cover different beats, not all write the same article".

---

### Part 3 — Why this is powerful (and why it's attractive for sEMG)

Summarizing:

**1. Channel-count/layout invariance**
- Different recordings can have 8, 16, 32 channels.
- LUNA's unification step always outputs **Q queries per patch**.
- The rest of the model doesn't care how many channels you had.

**2. Efficient computation**
- Attention operates over **patches**, not over every (channel,time) point.
- Cross-attention from Q queries to C channels is **linear in C**.
- This makes it feasible to handle high-density arrays.

**3. Good inductive bias for multi-channel biosignals**
- Per-channel time+frequency features (good for signals).
- Spatial info (coordinates) when available.
- Cross-channel integration via learned queries (rather than fixed pooling like average).

**4. Self-supervised pretraining**
- You don't need labels at scale.
- Very nice because EMG labels (pose, force) are expensive.

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

