"""
MoveR - Posture dataset collection from MoveR Digital Health and Care Hub.

Specs: Posture and movement recordings, organized by subject with subdirectory structure
Format: ZIP from GitHub (posture_dataset_collection/data/ folder with subject subfolders)
Size: N/A
"""
import os
import requests 
from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm

DATASET_NAME = "MoveR"

def download_mover(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://github.com/MoveR-Digital-Health-and-Care-Hub/posture_dataset_collection/archive/refs/heads/main.zip"
        zip_path = raw_dir / "mover.zip"
        
        print(f"  Downloading from GitHub...")
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
        
        # Move data folder to root level (keeping subfolders)
        print(f"  Organizing directory structure...")
        extracted_folder = raw_dir / "posture_dataset_collection-main" / "data"
        if extracted_folder.exists():
            for item in extracted_folder.iterdir():
                shutil.move(str(item), str(raw_dir / item.name))
            shutil.rmtree(raw_dir / "posture_dataset_collection-main")
        
        print(f"  Cleaning up archive file")
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_mover(data_root="/scratch/klambert/sEMG")

