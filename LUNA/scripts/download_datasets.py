import importlib

from download_data.BioPatRec import download_biopatrec
from download_data.CapgMyo import download_capgmyo
from download_data.emg2pose import download_emg2pose
from download_data.emg2qwerty import download_emg2qwerty
from download_data.GRABMyo import download_grabmyo
from download_data.MoveR import download_mover
from download_data.MyoKi import download_myoki
from download_data.Ninapro import download_ninapro
from download_data.putEMG import download_putemg
from download_data.SEEDS import download_seeds
typing_dataset = importlib.import_module("download_data.typing-dataset")
download_typing = typing_dataset.download_typing
from download_data.Zenodo import download_zenodo

CSL_HDEMG = importlib.import_module("download_data.CSL-HDEMG")
download_csl_hdemg = CSL_HDEMG.download_csl_hdemg

FORS_EMG = importlib.import_module("download_data.FORS-EMG")
download_fors_emg = FORS_EMG.download_fors_emg

HD_FW_KIN = importlib.import_module("download_data.HD-FW-KIN")
download_hd_fw_kin = HD_FW_KIN.download_hd_fw_kin

HD_sEMG = importlib.import_module("download_data.HD-sEMG")
download_hd_semg = HD_sEMG.download_hd_semg

multi_day = importlib.import_module("download_data.multi-day")
download_multi_day = multi_day.download_multi_day

muscle_fatigue = importlib.import_module("download_data.muscle-fatigue")
download_muscle_fatigue = muscle_fatigue.download_muscle_fatigue

UCI_EMG = importlib.import_module("download_data.UCI-EMG")
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

