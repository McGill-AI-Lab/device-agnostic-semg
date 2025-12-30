import os
import requests 
from pathlib import Path
import zipfile

DATASET_NAME = "Ninapro"

def download_ninapro(data_root = "./data"):
    try:
        dataset_root = Path(data_root) / DATASET_NAME
        raw_dir = dataset_root / "raw"
        preprocessed_dir = dataset_root / "preprocessed"
        
        # Create all directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # DB1: 27 subjects
        print("Downloading DB1...")
        db1_dir = raw_dir / "DB1"
        db1_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 28):
            subject_str = f"{subject_id:02d}"
            subject_dir = db1_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB1/Preprocessed/s{subject_id}.zip"
            zip_path = subject_dir / f"s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB1 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB1 subject {subject_str}: {e}")
        
        # DB2: 40 subjects
        print("Downloading DB2...")
        db2_dir = raw_dir / "DB2"
        db2_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 41):
            subject_str = f"{subject_id:02d}"
            subject_dir = db2_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{subject_id}.zip"
            zip_path = subject_dir / f"DB2_s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB2 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB2 subject {subject_str}: {e}")
        
        # DB3: 11 subjects
        print("Downloading DB3...")
        db3_dir = raw_dir / "DB3"
        db3_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 12):
            subject_str = f"{subject_id:02d}"
            subject_dir = db3_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB3/DB3_Preprocessed/S{subject_id}_A1_E1.zip"
            zip_path = subject_dir / f"S{subject_id}_A1_E1.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB3 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB3 subject {subject_str}: {e}")
        
        # DB4: 10 subjects
        print("Downloading DB4...")
        db4_dir = raw_dir / "DB4"
        db4_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 11):
            subject_str = f"{subject_id:02d}"
            subject_dir = db4_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB4/DB4_s{subject_id}.zip"
            zip_path = subject_dir / f"DB4_s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB4 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB4 subject {subject_str}: {e}")
        
        # DB5: 10 subjects
        print("Downloading DB5...")
        db5_dir = raw_dir / "DB5"
        db5_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 11):
            subject_str = f"{subject_id:02d}"
            subject_dir = db5_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB5/s{subject_id}.zip"
            zip_path = subject_dir / f"s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB5 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB5 subject {subject_str}: {e}")
        
        # DB6: 10 subjects
        print("Downloading DB6...")
        db6_dir = raw_dir / "DB6"
        db6_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 11):
            subject_str = f"{subject_id:02d}"
            subject_dir = db6_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB6/DB6_Preproc/DB6_s{subject_id}.zip"
            zip_path = subject_dir / f"DB6_s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB6 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB6 subject {subject_str}: {e}")
        
        # DB7: 22 subjects
        print("Downloading DB7...")
        db7_dir = raw_dir / "DB7"
        db7_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 23):
            subject_str = f"{subject_id:02d}"
            subject_dir = db7_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB7/DB7_Preproc/s{subject_id}.zip"
            zip_path = subject_dir / f"s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB7 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB7 subject {subject_str}: {e}")
        
        # DB8: 12 subjects
        print("Downloading DB8...")
        db8_dir = raw_dir / "DB8"
        db8_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 13):
            subject_str = f"{subject_id:02d}"
            subject_dir = db8_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB8/DB8_s{subject_id}.zip"
            zip_path = subject_dir / f"DB8_s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB8 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB8 subject {subject_str}: {e}")
        
        # DB9: 77 subjects
        print("Downloading DB9...")
        db9_dir = raw_dir / "DB9"
        db9_dir.mkdir(exist_ok=True)
        for subject_id in range(1, 78):
            subject_str = f"{subject_id:02d}"
            subject_dir = db9_dir / subject_str
            subject_dir.mkdir(exist_ok=True)
            
            url = f"https://ninapro.hevs.ch/files/DB9/s{subject_id}.zip"
            zip_path = subject_dir / f"s{subject_id}.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subject_dir)
                zip_path.unlink()
                print(f"  Downloaded DB9 subject {subject_str}")
            except Exception as e:
                print(f"  Failed DB9 subject {subject_str}: {e}")
        
        # DB10: Harvard Dataverse - requires manual download
        print("\nDB10 is hosted on Harvard Dataverse and requires manual download:")
        db10_dir = raw_dir / "DB10"
        db10_dir.mkdir(exist_ok=True)
        print(f"  Please download from these links and place in {db10_dir}:")
        print("  - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1Z3IOM")
        print("  - https://doi.org/10.7910/DVN/78QFZH")
        print("  - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F9R33N")
        print("  - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EJJ91H")
        
        print(f"\nDownloaded {DATASET_NAME} (DB1-DB9)")
        return raw_dir
        
    except Exception as e:
        print(f"Error downloading {DATASET_NAME}: {e}")
        return None

if __name__ == "__main__":
    download_ninapro()

