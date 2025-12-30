import os
import requests 
from pathlib import Path

DATASET_NAME = "FORS-EMG"

def download_fors_emg(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{DATASET_NAME} is a Kaggle dataset and requires Kaggle API or manual download.")
        print("Please download from: https://www.kaggle.com/datasets/ummerummanchaity/fors-emg-a-novel-semg-dataset/data")
        print(f"Place files in: {raw_dir}")
        print("\nTo use Kaggle API:")
        print("  1. Install: pip install kaggle")
        print("  2. Set up credentials: https://www.kaggle.com/docs/api")
        print("  3. Run: kaggle datasets download -d ummerummanchaity/fors-emg-a-novel-semg-dataset")
        
        return raw_dir
        
    except Exception as e:
        print(f"Error setting up {DATASET_NAME}: {e}")
        return None

