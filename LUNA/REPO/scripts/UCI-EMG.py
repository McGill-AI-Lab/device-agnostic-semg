import os
import requests 
from pathlib import Path

DATASET_NAME = "UCI-EMG"

def download_uci_emg(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # ============================================
        # DATASET-SPECIFIC DOWNLOAD LOGIC GOES HERE
        # ============================================
        # Example:
        # url = "https://example.com/dataset.zip"
        # response = requests.get(url, stream=True)
        # with open(raw_dir / "dataset.zip", "wb") as f:
        #     for chunk in response.iter_content(chunk_size=8192):
        #         f.write(chunk)
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

