"""
HD-sEMG - High-density surface EMG recordings of leg muscles from PhysioNet (v2.0.0).

Specs: High-density EMG electrode grid, leg muscle recordings, multiple electrode array
Format: ZIP archive from PhysioNet published project
Size: N/A
"""
import os
import requests 
from pathlib import Path
import zipfile

DATASET_NAME = "HD-sEMG"

def download_hd_semg(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://physionet.org/static/published-projects/hd-semg/high-density-surface-electromyography-recordings-of-the-leg-muscles-2.0.0.zip"
        zip_path = raw_dir / "hd-semg.zip"
        
        print(f"  Downloading from PhysioNet...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        print(f"  Cleaning up archive file")
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_hd_semg(data_root="/scratch/klambert/sEMG")

