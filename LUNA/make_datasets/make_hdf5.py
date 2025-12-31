"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            MAKE_HDF5.PY - PICKLE TO HDF5 CONVERSION UTILITY                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
   Converts preprocessed pickle files into efficient HDF5 format for fast I/O during
   training. Organizes data into hierarchical groups for optimal access patterns.

HIGH-LEVEL OVERVIEW:
   After running process_raw_eeg.py, you have thousands of pickle files. This script:
   1. Reads all pickle files from a directory
   2. Groups samples into batches (e.g., 1000 samples per group)
   3. Stores in HDF5 hierarchical structure
   4. Optionally removes pickle files to save disk space
   5. Handles corrupt files gracefully

KEY FUNCTIONS:
   
   1. create_hdf5(source_dir, target_file, finetune, group_size):
      - Iterates through pickle files
      - Collects samples into groups
      - Creates HDF5 datasets for X (signals) and y (labels)
      - Error handling for corrupt pickles
      
      Parameters:
      - source_dir: Directory with .pkl files
      - target_file: Output .h5 file path
      - finetune: Include labels (True) or just signals (False)
      - group_size: Samples per HDF5 group (default 1000)
   
   2. process_dataset(prepath, dataset_name, splits, finetune, remove_pkl):
      - Processes complete dataset (train/val/test splits)
      - Creates train.h5, val.h5, test.h5
      - Optionally removes processed/ pickle directory

HDF5 STRUCTURE:
   
   Output file structure:
   dataset.h5
   ├── data_group_0/
   │   ├── X: [1000, 22, 5120] float32  # EEG signals
   │   └── y: [1000] int64               # Labels (if finetune=True)
   ├── data_group_1/
   │   ├── X: [1000, 22, 5120]
   │   └── y: [1000]
   └── ...
   
   Where:
   - 22 = number of bipolar channels
   - 5120 = 20 seconds @ 256 Hz
   - Groups enable partial loading (don't need entire file in RAM)

WHY HDF5?
   
   Advantages over pickles:
   ✓ Random access: Load any sample without reading entire file
   ✓ Compression: Efficient storage with gzip/lzf
   ✓ Partial loading: Only load needed groups
   ✓ Memory-mapped I/O: OS manages caching
   ✓ Cross-platform: Works on Linux/Windows/Mac
   
   Disadvantages of pickles:
   ✗ Sequential access: Must unpickle entire file
   ✗ Slow for large datasets
   ✗ No compression built-in
   ✗ Many small files → filesystem overhead

TYPICAL WORKFLOW:
   
   Step 1: Process raw EDF → pickles
   ```bash
   python process_raw_eeg.py tusl --root_dir /eeg_data/TUSL/edf --output_dir /processed_eeg
   ```
   
   Step 2: Bundle pickles → HDF5
   ```bash
   python make_hdf5.py --prepath /processed_eeg --dataset TUSL --remove_pkl
   ```
   
   Output:
   /processed_eeg/TUSL_data/
   ├── train.h5  (e.g., 2.5 GB, 50k samples)
   ├── val.h5    (e.g., 500 MB, 10k samples)
   └── test.h5   (e.g., 500 MB, 10k samples)

USAGE:
   ```bash
   # Process all datasets
   python make_hdf5.py --prepath /processed_eeg --dataset All --remove_pkl
   
   # Process only TUAB
   python make_hdf5.py --prepath /processed_eeg --dataset TUAB
   ```

RELATED FILES:
   - make_datasets/process_raw_eeg.py: Creates the pickle files
   - datasets/hdf5_dataset.py: Loads the HDF5 files during training
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
#*----------------------------------------------------------------------------*
import os
import pickle
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import shutil

def create_hdf5(source_dir, target_file, finetune=True,group_size=1000):
    """
    Creates an HDF5 file from a directory of pickle files.
    Includes error handling for corrupt files.
    """
    files = os.listdir(source_dir)
    data_group = []
    # Filter out non-pickle files
    files = [f for f in files if f.endswith('.pkl')]
    
    with h5py.File(target_file, 'w') as h5f:
        for i, file in enumerate(tqdm(files, desc=f"Creating {target_file}")):
            with open(os.path.join(source_dir, file), 'rb') as f:
                try:
                    sample = pickle.load(f)
                    data_group.append(sample)
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"Warning: Skipping corrupt pickle file {file}: {e}")
                    continue
                
                if (i + 1) % group_size == 0 or i == len(files) - 1:
                    if not data_group:
                        continue
                    
                    grp = h5f.create_group(f"data_group_{i // group_size}")
                    
                    try:
                        X_data = np.array([s['X'] for s in data_group])
                        grp.create_dataset("X", data=X_data)
                    except KeyError:
                        print(f"Error: 'X' key missing in data group {i // group_size}. Skipping group.")
                        del h5f[f"data_group_{i // group_size}"]
                        data_group = []
                        continue
                    except Exception as e:
                        print(f"Error packing X data for group {i // group_size}: {e}")
                        del h5f[f"data_group_{i // group_size}"]
                        data_group = []
                        continue

                    if(finetune):
                        try:
                            y_data = np.array([s['y'] for s in data_group])
                            grp.create_dataset("y", data=y_data)
                        except KeyError:
                            print(f"Error: 'y' key missing in finetune mode for group {i // group_size}. Skipping group.")
                            del h5f[f"data_group_{i // group_size}"]
                        except Exception as e:
                             print(f"Error packing y data for group {i // group_size}: {e}")
                             del h5f[f"data_group_{i // group_size}"]
                    
                    data_group = []

def process_dataset(prepath, dataset_name, splits, finetune, remove_pkl):
    """
    Helper function to process a single dataset.
    prepath: The root directory for data
    dataset_name: e.g., 'TUAR_data'
    splits: list of splits, e.g., ['train', 'val']
    finetune: boolean, whether to save 'y' labels
    remove_pkl: boolean, whether to delete the processed pkl directory
    """
    print(f"--- Processing {dataset_name} ---")
    
    # Path to the directory containing all split folders (train/, val/, test/)
    processed_dir_path = os.path.join(prepath, dataset_name, "processed")

    for td in splits:
        target_h5_file = os.path.join(prepath, dataset_name, f"{td}.h5")
        source_pickle_dir = os.path.join(processed_dir_path, td) # Use processed_dir_path

        if os.path.exists(target_h5_file):
            print(f"{dataset_name} {td}.h5 already exists. Skipping...")
        elif not os.path.isdir(source_pickle_dir):
            print(f"Source directory not found: {source_pickle_dir}")
            print(f"Skipping {dataset_name} {td}.h5")
        else:
            # Ensure target directory exists
            os.makedirs(os.path.dirname(target_h5_file), exist_ok=True)
            create_hdf5(source_pickle_dir, target_h5_file, finetune=finetune)

    # After processing all splits for this dataset, remove the pkl directory if requested
    if remove_pkl:
        if os.path.isdir(processed_dir_path):
            print(f"Removing processed .pkl directory: {processed_dir_path}")
            try:
                shutil.rmtree(processed_dir_path)
                print(f"Successfully removed {processed_dir_path}")
            except Exception as e:
                print(f"Error removing {processed_dir_path}: {e}")
        else:
            print(f"Processed directory not found, cannot remove: {processed_dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HDF5 files from processed .pkl files.")
    parser.add_argument(
        "--prepath",
        type=str,
        required=True,
        help="The root directory containing the processed dataset folders (e.g., TUAR_data, TUSL_data)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="All",
        choices=["TUAR", "TUSL", "TUAB", "All"],
        help="Which dataset to process. 'All' processes all three."
    )
    parser.add_argument(
        "--remove_pkl",
        action="store_true",
        help="If set, removes the 'processed' directory (containing .pkl files) after HDF5 creation."
    )
    args = parser.parse_args()

    # Define the splits we expect for ALL datasets
    all_splits = ['train', 'val', 'test']
    
    datasets_to_process = []
    if args.dataset == "All":
        datasets_to_process = ["TUAR_data", "TUSL_data", "TUAB_data"]
    elif args.dataset == "TUAR":
        datasets_to_process = ["TUAR_data"]
    elif args.dataset == "TUSL":
        datasets_to_process = ["TUSL_data"]
    elif args.dataset == "TUAB":
        datasets_to_process = ["TUAB_data"]

    # Loop through the selected datasets and process them
    for data_folder_name in datasets_to_process:
        process_dataset(
            args.prepath,
            dataset_name=data_folder_name,
            splits=all_splits,
            finetune=True,
            remove_pkl=args.remove_pkl
        )

    print("HDF5 creation complete.")