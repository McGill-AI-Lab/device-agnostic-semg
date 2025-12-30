"""
putEMG - EMG dataset from Poznań University of Technology BioLab with 44 subjects, 7 gestures.

Specs: Multiple electrode configurations, repeated trials, standardized acquisition protocol
Format: Individual HDF5 files (emg_gestures-{subject:02d}-{gesture:03d}.hdf5)
Size: N/A per file (44 subjects × 7 gestures = ~308 individual HDF5 files)
"""
import os
import requests 
from pathlib import Path

DATASET_NAME = "putEMG"

def download_putemg(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        base_url = "https://biolab.put.poznan.pl/putemg-dataset/data"
        
        # Dataset structure: Data-HDF5 format organized by subject and gesture
        data_types = ["Data-HDF5"]
        
        print(f"  Downloading individual HDF5 files (44 subjects × 7 gestures)...")
        for data_type in data_types:
            data_dir = raw_dir / data_type
            data_dir.mkdir(exist_ok=True)
            for subject_id in range(1, 45):  # 44 subjects in putEMG
                subject_str = f"{subject_id:02d}"
                
                for gesture_id in range(1, 8):  # 7 gestures
                    filename = f"emg_gestures-{subject_str}-{gesture_id:03d}.hdf5"
                    file_url = f"{base_url}/{data_type}/{filename}"
                    file_path = data_dir / filename
                    
                    try:
                        response = requests.get(file_url, stream=True)
                        if response.status_code == 200:
                            with open(file_path, "wb") as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"    Downloaded: {filename}")
                        elif response.status_code == 404:
                            # File doesn't exist, skip
                            continue
                    except Exception as e:
                        print(f"    Skipped {filename}: {e}")
                        continue
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_putemg(data_root="/scratch/klambert/sEMG")

