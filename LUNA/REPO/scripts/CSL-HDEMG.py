"""
CSL-HDEMG - High-density EMG dataset (currently unavailable).

Specs: N/A (dataset unavailable)
Format: N/A
Size: N/A
"""
import os
import requests 
from pathlib import Path

DATASET_NAME = "CSL-HDEMG"

def download_csl_hdemg(data_root = "./data"):
    print(f"Starting download for {DATASET_NAME}")
    print(f"  {DATASET_NAME} is currently unavailable.")
    pass # unavailable
    # try:
    #     dataset_root = Path(data_root) / DATASET_NAME
    #     raw_dir = dataset_root / "raw"
    #     preprocessed_dir = dataset_root / "preprocessed"
    #     
    #     # Create all directories
    #     raw_dir.mkdir(parents=True, exist_ok=True)
    #     preprocessed_dir.mkdir(parents=True, exist_ok=True)
    #     
    #     # ============================================
    #     # DATASET-SPECIFIC DOWNLOAD LOGIC GOES HERE
    #     # ============================================
    #     # Example:
    #     # url = "https://example.com/dataset.zip"
    #     # response = requests.get(url, stream=True)
    #     # with open(raw_dir / "dataset.zip", "wb") as f:
    #     #     for chunk in response.iter_content(chunk_size=8192):
    #     #         f.write(chunk)
    #     
    #     print(f"Downloaded {DATASET_NAME}")
    #     return raw_dir
    #     
    # except Exception as e:
     #     print(f"Error downloading {DATASET_NAME}: {e}")
     #     return None

if __name__ == "__main__":
    download_csl_hdemg(data_root="/scratch/klambert/sEMG")
