Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Datasets

This directory contains the PyTorch `Dataset` classes for the project. In PyTorch, a `Dataset` class is responsible for storing and providing access to the samples in your data (e.g., EEG/EMG signals and their corresponding labels). This directory defines how individual data points are loaded and processed.

**Now supports both EEG and sEMG datasets!**

---

## The HDF5Loader

To handle the large-scale EEG datasets used in our experiments, we use a custom `HDF5Loader` class defined in `hdf5_dataset.py`. Storing data in the HDF5 format is highly efficient for I/O operations, which is crucial when working with datasets that can be several gigabytes in size.

**Key Features:**
-   **Efficient Loading**: Reads data directly from `.h5` files, which is significantly faster than loading thousands of individual files.
-   **Memory Caching**: Includes an optional caching mechanism to store recently accessed samples in memory, further speeding up data loading during training.
-   **Flexible Modes**: Supports both `finetune` mode (loading both signals and labels) and pre-training mode (loading signals only).
-   **Grouped Data**: Assumes data within the HDF5 file is organized into groups, which helps manage large datasets.

For details on how to convert raw EEG data into the required HDF5 format, please see the documentation in the [`../make_datasets`](../make_datasets) directory.

---

## Supported Datasets

Our framework is primarily designed to work with several of the Temple University Hospital (TUH) EEG corpora. For more detailed information on each, please refer to the documentation in the [`../docs/datasets`](../docs/datasets) directory.

### 1. **TUH Abnormal EEG (TUAB) Dataset**
-   **Purpose**: Used for classifying EEG sessions as either "normal" or "abnormal".
-   **Size**: Contains over 3,000 EEG sessions.
-   **Task**: Binary classification.

### 2. **TUH Artifact (TUAR) Dataset**
-   **Purpose**: Designed for detecting various types of artifacts in EEG signals.
-   **Annotations**: Covers five common artifact types, including eye movement, chewing, and electrode noise.
-   **Tasks**: Supports multiple classification protocols such as binary, multi-label, and multi-class classification.

### 3. **TUH Slowing (TUSL) Dataset**
-   **Purpose**: Curated for classifying different types of slowing events in EEG signals, which can be indicative of neurological disorders.
-   **Annotations**: Includes four classes: slowing, seizure, complex background, and normal.
-   **Task**: Multi-class classification.

### 4. **TUH EEG (TUEG) Dataset**
-   **Purpose**: This is our primary dataset for self-supervised pre-training.
-   **Size**: One of the largest publicly available EEG corpora, containing over 21,000 hours of recordings from more than 14,000 patients.
-   **Usage**: We use this large, diverse dataset to train our foundation models to learn robust representations of EEG signals before fine-tuning them on specific downstream tasks.

### 5. SEED-V Dataset
- **Purpose**: EEG-based emotion recognition with video-elicited stimuli.
- **Size**: 15 subjects with 15 sessions. Includes 62 electrodes.
- **Tasks**: Multi-class classification, 5 classes for emotions in SEED-V.
- **Notes**: Usually used with subject-independent evaluation; the subject_independent_data_module is supported for this dataset.

### 6. **Siena Dataset**
-   **Purpose**: The Siena dataset is used for self-supervised pre-training.
-   **Size**: 14 subjects and 141 hours of EEG data. Includes 29 electrodes.
-   **Usage**: We use this dataset to have different electrode configurations in out pre-training data, in addition to TUEG dataset.

---

## sEMG (Surface Electromyography) Datasets

The `semg/` subdirectory contains dataset loaders for surface EMG data. Unlike EEG, sEMG datasets:
- Have variable channel counts (4-320 channels)
- Support both **gesture classification** AND **pose regression**
- Don't have standardized electrode placements
- Often include kinematic labels (glove DOFs, forces)

### sEMG Dataset Loaders

#### 1. **sEMGHDF5Dataset** (`semg/semg_hdf5_dataset.py`)
Generic HDF5 loader for sEMG data supporting multiple task types:
- `mode='pretrain'`: Self-supervised (no labels)
- `mode='classify'`: Gesture classification
- `mode='regress'`: Pose regression (joint angles)

#### 2. **NinaproDataset** (`semg/ninapro_dataset.py`)
Loader for the Ninapro database family, the most widely used sEMG dataset:

| Database | Channels | Subjects | Gestures | Sampling Rate | Hardware |
|----------|----------|----------|----------|---------------|----------|
| DB1 | 10 | 27 | 52 | 100 Hz | Otto Bock MyoBock |
| DB2 | 12 | 40 | 49 | 2000 Hz | Delsys Trigno |
| DB3 | 12 | 11 | 49 | 2000 Hz | Delsys (amputees) |
| DB4 | 12 | 10 | 52 | 2000 Hz | Cometa |
| DB5 | 16 | 10 | 52 | 200 Hz | 2× Myo armband |
| DB6 | 14 | 10 | 7 | 2000 Hz | Delsys (multi-day) |
| DB7 | 12 | 20 | 40 | 2000 Hz | Delsys + IMU |
| DB8 | 16 | 10 | 9 | 2000 Hz | Delsys (finger regression) |

#### 3. **NinaproPoseDataset** (`semg/ninapro_dataset.py`)
For pose regression using CyberGlove data (DB1, DB2, DB4, DB5, DB7, DB8):
- Returns continuous joint angles (18-22 DOF)
- Perfect for EMG → hand pose regression tasks

### sEMG Sample Format

All sEMG datasets return dict-style samples:
```python
{
    'input': tensor [C, T],           # EMG signal
    'label': tensor or None,          # Gesture class (classification)
    'pose': tensor [D] or None,       # Joint angles (regression)
    'subject_id': int,                # For subject-independent splits
    'num_channels': int,              # For channel-agnostic batching
    'dataset_id': str,                # Source dataset identifier
}
```

### Preprocessing sEMG Data

To preprocess raw Ninapro data into HDF5 format:
```bash
python make_datasets/process_ninapro.py \
    --db db2 \
    --root_dir /data/ninapro/db2 \
    --output_dir /processed/ninapro \
    --mode classify  # or 'regress' for pose regression
```

---

## Recommended sEMG Datasets by Task

### For Pose Regression (EMG → Hand Kinematics)
Best datasets with continuous joint angles:
- **emg2pose** (Meta): 16ch wristband, full 3D hand, 370 hours
- **Ninapro DB7/DB8**: Designed for finger regression
- **Ninapro DB1/2/4/5**: 22-DOF CyberGlove
- **SEEDS**: 134ch HD + 18-DOF glove

### For Channel-Agnostic Training
Mix these datasets to learn representations that work across channel counts:
- Train on HD (64+ channels): CapgMyo, CSL-HDEMG, SEEDS
- Validate on mid-channel (10-16): DB2, DB5, GRABMyo
- Test on low-channel (4-8): UCI EMG, FORS-EMG

### For Cross-Device Evaluation
Same gestures, different hardware:
- DB1 (MyoBock, 10ch) ↔ DB4 (Cometa, 12ch) ↔ DB5 (Myo, 16ch)

### For Multi-Day/Temporal Drift
- DB6: 5 days × 2 sessions
- GRABMyo: 3 days (Day 1, 8, 29)