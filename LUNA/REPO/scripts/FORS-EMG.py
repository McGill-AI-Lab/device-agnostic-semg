"""
FORS-EMG - Novel sEMG dataset from Kaggle (requires manual download or Kaggle API).

Specs: Surface EMG recordings for gesture classification tasks
Format: Kaggle dataset format (various file types)
Size: N/A (manual download required)
"""
import os
import requests 
from pathlib import Path

DATASET_NAME = "FORS-EMG"

def download_fors_emg(data_root = "./data"):
    try:
        print(f"Starting setup for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  {DATASET_NAME} is a Kaggle dataset and requires Kaggle API or manual download.")
        print("  Please download from: https://www.kaggle.com/datasets/ummerummanchaity/fors-emg-a-novel-semg-dataset/data")
        print(f"  Place files in: {raw_dir}")
        print("\n  To use Kaggle API:")
        print("    1. Install: pip install kaggle")
        print("    2. Set up credentials: https://www.kaggle.com/docs/api")
        print("    3. Run: kaggle datasets download -d ummerummanchaity/fors-emg-a-novel-semg-dataset")
        
        return raw_dir
        
    except Exception as e:
        print(f"Error setting up {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_fors_emg(data_root="/scratch/klambert/sEMG")

