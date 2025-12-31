"""
GRABMyo - Dataset of complex hand grasps with EMG from PhysioNet (v1.1.0).

Specs: EMG recordings during various complex hand grasp tasks, multiple grasp types/patterns
Format: ZIP archive from PhysioNet published project
Size: N/A
"""
import os
import requests 
from pathlib import Path
import zipfile
from tqdm import tqdm

DATASET_NAME = "GRABMyo"

def download_grabmyo(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://physionet.org/static/published-projects/grabmyo/grabmyo-a-dataset-of-complex-hand-grasps-1.1.0.zip"
        zip_path = raw_dir / "grabmyo.zip"
        
        print(f"  Downloading from PhysioNet...")
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
    download_grabmyo(data_root="/scratch/klambert/sEMG")

