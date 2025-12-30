"""
BioPatRec - EMG pattern recognition dataset from BioPatRec project's Data_Repository branch.

Specs: EMG data for pattern recognition in prosthetic control applications
Format: ZIP from GitHub Data_Repository branch with organized dataset structure
Size: N/A (varies by repository contents)
"""
import os
import requests 
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

DATASET_NAME = "BioPatRec"

def download_biopatrec(data_root = "./data"):
    try:
        print(f"Starting download for {DATASET_NAME}")
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        print(f"  Creating directories at {dataset_root}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://github.com/biopatrec/biopatrec/archive/refs/heads/Data_Repository.zip"
        zip_path = raw_dir / "Data_Repository.zip"
        
        print(f"  Downloading from GitHub...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, "wb") as f, tqdm(
                desc="    Progress",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"  Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

        # Move zip archive contents up into raw_dir
        print(f"  Organizing directory structure...")
        top_dirs = [
            p for p in raw_dir.iterdir()
            if p.is_dir() and p.name.startswith("biopatrec-Data_Repository")
        ]
        if len(top_dirs) == 1:
            extracted_root = top_dirs[0]
            for item in extracted_root.iterdir():
                dest = raw_dir / item.name
                if dest.exists():
                    continue  # don't clobber anything already there
                shutil.move(str(item), str(dest))
            shutil.rmtree(extracted_root, ignore_errors=True)

        # cleanup
        print(f"  Cleaning up archive file")
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_biopatrec(data_root="/scratch/klambert/sEMG")
