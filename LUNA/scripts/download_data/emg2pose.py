"""
emg2pose - Large-scale EMG to full-body pose estimation dataset from Facebook Research.

Specs: Synchronized EMG and full-body pose/motion data for movement prediction tasks
Format: TAR archive containing dataset files
Size: 431 GiB (very large) ~600 MiB (mini)
"""
import os
import requests 
from pathlib import Path
import tarfile
from tqdm import tqdm

DATASET_NAME = "emg2pose"
MINI = True

def download_emg2pose(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME} ({'mini' if MINI else 'full'} version)")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        if MINI:
            url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"
            tar_path = raw_dir / "emg2pose_dataset_mini.tar"
        else:
            url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
            tar_path = raw_dir / "emg2pose_dataset.tar"
        
        print(f"  Downloading from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, "wb") as f, tqdm(
            desc="    Progress",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"  Extracting archive...")
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(raw_dir)
        
        print(f"  Cleaning up archive file")
        tar_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_emg2pose(data_root="/scratch/klambert/sEMG")

