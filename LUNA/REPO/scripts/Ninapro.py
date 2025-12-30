import os
import requests 
from pathlib import Path

DATASET_NAME = "Ninapro"

def download_ninapro(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        

        # Create subfolders for DB1-10
        for i in range(1, 11):
            db_dir = raw_dir / f"DB{i}"
            db_dir.mkdir(exist_ok=True)
        
        print(f"{DATASET_NAME} structure created. Please manually download from https://ninapro.hevs.ch/")
        print("Place each database in its corresponding DB1-DB10 subfolder.")
        
        return raw_dir
        
    except Exception as e:
        print(f"Error setting up {DATASET_NAME}: {e}")
        return None

