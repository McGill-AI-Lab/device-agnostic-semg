"""
CapgMyo - High-density sEMG for gesture recognition using instantaneous EMG images (CapgMyo-DBa).

Specs: High-density EMG electrode array, multiple hand gestures, gesture recognition tasks
Format: ZIP archive from Figshare containing dataset files
Size: N/A
"""
import os
import requests 
from pathlib import Path
import zipfile

DATASET_NAME = "CapgMyo"

def download_capgmyo(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://figshare.com/ndownloader/articles/7210397/versions/2"
        zip_path = raw_dir / "capgmyo.zip"
        
        print(f"  Downloading from Figshare...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract zip
        print(f"  Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        # Clean up zip
        print(f"  Cleaning up archive file")
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_capgmyo(data_root="/scratch/klambert/sEMG")

