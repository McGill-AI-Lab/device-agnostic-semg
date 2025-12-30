import os
import requests 
from pathlib import Path
import tarfile

DATASET_NAME = "emg2qwerty"

def download_emg2qwerty(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://research.facebook.com/file/1030086308123356/emg2qwerty-dataset.tar.gz"
        tar_path = raw_dir / "emg2qwerty-dataset.tar.gz"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(raw_dir)
        
        # Clean up tar.gz
        tar_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_emg2qwerty()

