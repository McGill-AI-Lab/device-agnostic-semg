"""
MyoKi (Myo Dataset) - sEMG data from 13 subjects performing Rock-Paper-Scissors gestures with Myo Armband.

Specs: 8-channel sEMG (sensors 1-8 around forearm + 2 on flexor/extensor digitorum), 
       10 repetitions per gesture, 2-second recordings with 0.5s rest between
Format: ZIP containing CSV files (emg.csv, pose.csv, orientation.csv, meta.csv per recording)
Size: N/A (individual files ~400 entries each)
"""
import os
import requests 
from pathlib import Path
import zipfile
import shutil

DATASET_NAME = "MyoKi"

def download_myoki(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://github.com/michidk/myo-dataset/archive/refs/heads/main.zip"
        zip_path = raw_dir / "myo-dataset.zip"
        
        print(f"  Downloading from GitHub...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract zip
        print(f"  Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        # Move dataset folder to root level
        print(f"  Organizing directory structure...")
        extracted_folder = raw_dir / "myo-dataset-main" / "dataset"
        if extracted_folder.exists():
            for item in extracted_folder.iterdir():
                shutil.move(str(item), str(raw_dir / item.name))
            shutil.rmtree(raw_dir / "myo-dataset-main")
        
        # Clean up zip
        print(f"  Cleaning up archive file")
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_myoki(data_root="/scratch/klambert/sEMG")

