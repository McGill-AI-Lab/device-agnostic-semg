import importlib

from BioPatRec import download_biopatrec
from CapgMyo import download_capgmyo
from emg2pose import download_emg2pose
from emg2qwerty import download_emg2qwerty
from GRABMyo import download_grabmyo
from MoveR import download_mover
from MyoKi import download_myoki
from Ninapro import download_ninapro
from putEMG import download_putemg
from SEEDS import download_seeds
from typing import download_typing
from Zenodo import download_zenodo

CSL_HDEMG = importlib.import_module("CSL-HDEMG")
download_csl_hdemg = CSL_HDEMG.download_csl_hdemg

FORS_EMG = importlib.import_module("FORS-EMG")
download_fors_emg = FORS_EMG.download_fors_emg

HD_FW_KIN = importlib.import_module("HD-FW-KIN")
download_hd_fw_kin = HD_FW_KIN.download_hd_fw_kin

HD_sEMG = importlib.import_module("HD-sEMG")
download_hd_semg = HD_sEMG.download_hd_semg

multi_day = importlib.import_module("multi-day")
download_multi_day = multi_day.download_multi_day

muscle_fatigue = importlib.import_module("muscle-fatigue")
download_muscle_fatigue = muscle_fatigue.download_muscle_fatigue

UCI_EMG = importlib.import_module("UCI-EMG")
download_uci_emg = UCI_EMG.download_uci_emg

DATA_ROOT = "/scratch/klambert/sEMG"

def main():
    """Download all EMG datasets to the specified data directory."""
    download_functions = [
        download_biopatrec,
        download_capgmyo,
        download_csl_hdemg,
        download_emg2pose,
        download_emg2qwerty,
        download_fors_emg,
        download_grabmyo,
        download_hd_fw_kin,
        download_hd_semg,
        download_mover,
        download_multi_day,
        download_muscle_fatigue,
        download_myoki,
        download_ninapro,
        download_putemg,
        download_seeds,
        download_typing,
        download_uci_emg,
        download_zenodo,
    ]
    
    print(f"Downloading {len(download_functions)} datasets...")
    for download_func in download_functions:
        download_func(DATA_ROOT)
    
    print("\nAll downloads complete!")

if __name__ == "__main__":
    main()

