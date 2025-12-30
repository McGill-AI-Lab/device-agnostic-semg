import os
import requests 
import zipfile
import shutil
from pathlib import Path

DATASET_NAME = "BioPatRec"

def download_biopatrec(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://github.com/biopatrec/biopatrec/archive/refs/heads/Data_Repository.zip"
        zip_path = raw_dir / "Data_Repository.zip"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

        # Move zip archive contents up into raw_dir
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
        zip_path.unlink()
        
        print(f"Downloaded {DATASET_NAME}")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_biopatrec()
