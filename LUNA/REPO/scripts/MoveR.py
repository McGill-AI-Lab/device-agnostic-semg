import os
import requests 
from pathlib import Path
import zipfile
import shutil

DATASET_NAME = "MoveR"

def download_mover(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://github.com/MoveR-Digital-Health-and-Care-Hub/posture_dataset_collection/archive/refs/heads/main.zip"
        zip_path = raw_dir / "mover.zip"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        # Move data folder to root level (keeping subfolders)
        extracted_folder = raw_dir / "posture_dataset_collection-main" / "data"
        if extracted_folder.exists():
            for item in extracted_folder.iterdir():
                shutil.move(str(item), str(raw_dir / item.name))
            shutil.rmtree(raw_dir / "posture_dataset_collection-main")
        
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

