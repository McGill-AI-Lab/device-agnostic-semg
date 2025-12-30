"""
muscle-fatigue - sEMG recordings during muscle fatigue conditions from Zenodo (record 14182446).

Specs: sEMG data collected under muscle fatigue scenarios, temporal progression of fatigue
Format: ZIP archive (sEMG_data.zip) containing data files and analysis scripts
Size: N/A
"""
import os
import requests 
from pathlib import Path
import zipfile
from tqdm import tqdm

DATASET_NAME = "muscle-fatigue"

def download_muscle_fatigue(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://zenodo.org/records/14182446/files/sEMG_data.zip?download=1"
        zip_path = raw_dir / "muscle-fatigue.zip"
        
        print(f"  Downloading from Zenodo...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, "wb") as f, tqdm(
            desc="    Progress",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
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
    download_muscle_fatigue(data_root="/scratch/klambert/sEMG")

