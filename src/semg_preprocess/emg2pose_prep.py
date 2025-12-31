from pathlib import Path
from typing import Literal, Tuple, List

import h5py
import numpy as np
import pandas as pd

from .signals import PreprocessConfig, preprocess_continuous_emg
from .windowing import make_emg_pose_windows
from .utils import load_config

Split = Literal["train", "val", "test"]

class Emg2PosePreprocessor:
    """
    Handles:
      - reading metadata.csv
      - iterating over HDF5 session files
      - running generic preprocessing
      - windowing and saving to a single HDF5 per split
    """

    def __init__(self, config_path: Path, cfg: PreprocessConfig):
        config = load_config(config_path)
        
        # Get data_root from config
        data_root = Path(config['paths']['data_root'])
        
        # Set up paths: output_dir = data_root/emg2pose/preprocessed
        self.output_dir = data_root / "emg2pose" / "preprocessed"
        self.metadata_csv = self.output_dir / "metadata.csv"
        self.data_root = data_root  # for loading raw data files
        
        self.cfg = cfg

    def load_session(self, h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single EMG2Pose session file.

        Expected:
          - emg: [T, C]
          - pose: [T, D]
        """
        with h5py.File(h5_path, "r") as f:
            # TODO: inspect one file to confirm these dataset names:
            # print(list(f.keys())) in a small debug script.
            emg = f["emg"][...]      # shape [T, C] or [C, T] depending on dataset
            pose = f["pose"][...]    # shape [T, D]

        # If emg comes as [C, T], transpose it here.
        if emg.shape[0] < emg.shape[1]:  # heuristic; adjust if needed
            emg = emg.T

        return emg, pose

    def _collect_split_rows(self, split: Split) -> pd.DataFrame:
        meta = pd.read_csv(self.metadata_csv)

        # EMG2Pose metadata columns include:
        # user, session, stage, side, moving_hand, held_out_user,
        # held_out_stage, split, generalization, ... :contentReference[oaicite:1]{index=1}
        split_meta = meta[meta["split"] == split].copy()

        return split_meta

    def _get_h5_path_for_row(self, row: pd.Series) -> Path:
        """
        Convert metadata row -> full path to HDF5 file.

        You need to look at metadata.csv once to see how paths are stored.
        Common patterns:
          - a 'filename' column with something like 'user123_session4_stage01_left.h5'
          - or a 'path' column with relative paths.

        For now we assume there is a 'filename' column and files live
        directly under data_root. Adjust as needed.
        """
        filename_col = "filename"  # TODO: change to real column name
        filename = row[filename_col]
        return self.data_root / filename

    def _process_single_file(
        self,
        h5_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load, preprocess, and window a single session file.
        """
        emg_raw, pose_raw = self.load_session(h5_path)

        # 1) Preprocess EMG (resample, filter, normalize)
        emg_proc, _stats = preprocess_continuous_emg(
            emg_raw,
            cfg=self.cfg,
        )

        # 2) (Optional) resample pose to same fs_target if needed.
        # For EMG2Pose, EMG and pose are time-aligned at 2kHz already,
        # but if fs changes, you would need to resample pose too.

        # 3) Window into fixed-length segments
        # Example: 200 ms windows, 50 ms stride
        X, y = make_emg_pose_windows(
            emg=emg_proc,
            pose=pose_raw,
            fs=self.cfg.fs_target,
            window_ms=200.0,
            stride_ms=50.0,
            label_mode="center",
        )

        return X, y

    def build_split(self, split: Split) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build all windows for a given split and return them as big arrays.
        """
        split_rows = self._collect_split_rows(split)

        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        for _, row in split_rows.iterrows():
            h5_path = self._get_h5_path_for_row(row)
            X, y = self._process_single_file(h5_path)

            if X.shape[0] == 0:
                continue

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            return (
                np.zeros((0, 1, 1), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )

        X_all = np.concatenate(all_X, axis=0)  # [N_total, T_win, C]
        y_all = np.concatenate(all_y, axis=0)  # [N_total, D]

        return X_all, y_all

    def save_split(self, split: Split) -> Path:
        """
        Build split and save to HDF5 as:
          - 'emg': [N, T_win, C]
          - 'pose': [N, D]
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"emg2pose_{split}.h5"

        X_all, y_all = self.build_split(split)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("emg", data=X_all, compression="gzip")
            f.create_dataset("pose", data=y_all, compression="gzip")

        return out_path

