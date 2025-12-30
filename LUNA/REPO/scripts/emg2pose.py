import os
import requests 
from pathlib import Path
import tarfile

DATASET_NAME = "emg2pose"

def download_emg2pose(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
        tar_path = raw_dir / "emg2pose_dataset.tar"
        
        print(f"Downloading {DATASET_NAME}...")
        print("WARNING: This is a large dataset (431 GiB).")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (1024 * 1024 * 100) == 0:  # Print every 100MB
                    progress = (downloaded / total_size) * 100
                    print(f"  Progress: {downloaded / (1024**3):.2f} GiB / {total_size / (1024**3):.2f} GiB ({progress:.1f}%)")
        
        print(f"Download complete. Extracting...")
        
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(raw_dir)
        
        tar_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

