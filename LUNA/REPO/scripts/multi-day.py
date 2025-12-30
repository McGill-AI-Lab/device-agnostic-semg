"""
multi-day - Longitudinal EMG dataset across multiple days from Zenodo (2 parts: records 15077957, 15070187).

Specs: Multi-day EMG recordings for studying signal stability and variability over time
Format: Two separate ZIP archives (Data.zip for part1 and part2)
Size: N/A per part
"""
import os
import requests 
from pathlib import Path
import zipfile

DATASET_NAME = "multi-day"

def download_multi_day(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url_part1 = "https://zenodo.org/records/15077957/files/Data.zip?download=1"
        zip_path1 = raw_dir / "multi-day-part1.zip"
        
        print(f"  Downloading part 1 from Zenodo...")
        response1 = requests.get(url_part1, stream=True)
        response1.raise_for_status()
        
        with open(zip_path1, "wb") as f:
            for chunk in response1.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  Extracting part 1...")
        with zipfile.ZipFile(zip_path1, 'r') as zip_ref:
            zip_ref.extractall(raw_dir / "part1")
        
        zip_path1.unlink()
        
        url_part2 = "https://zenodo.org/records/15070187/files/Data.zip?download=1"
        zip_path2 = raw_dir / "multi-day-part2.zip"
        
        print(f"  Downloading part 2 from Zenodo...")
        response2 = requests.get(url_part2, stream=True)
        response2.raise_for_status()
        
        with open(zip_path2, "wb") as f:
            for chunk in response2.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  Extracting part 2...")
        with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
            zip_ref.extractall(raw_dir / "part2")
        
        print(f"  Cleaning up archive files")
        zip_path2.unlink()
        
        print(f"Downloaded {DATASET_NAME} (both parts)")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_multi_day(data_root="/scratch/klambert/sEMG")

