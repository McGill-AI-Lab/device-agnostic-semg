import os
import requests 
from pathlib import Path
import subprocess
import getpass

DATASET_NAME = "HD-FW-KIN"

def download_hd_fw_kin(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{DATASET_NAME} requires a PhysioNet account and license agreement.")
        print("Please create an account at: https://physionet.org/")
        print("Then sign the data use agreement for this dataset.")
        
        username = input("Enter PhysioNet username: ")
        password = getpass.getpass("Enter PhysioNet password: ")
        
        url = "https://physionet.org/files/hand-kinematics-semg/1.0.0/"
        
        result = subprocess.run([
            "wget", "-r", "-N", "-c", "-np",
            "--user", username,
            "--password", password,
            "-P", str(raw_dir),
            url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Downloaded {DATASET_NAME}")
            return raw_dir
        else:
            print(f"Error: {result.stderr}")
            return None
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_hd_fw_kin()

