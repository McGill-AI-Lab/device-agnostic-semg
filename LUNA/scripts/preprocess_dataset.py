from pathlib import Path
import argparse

from src.semg_preproc.emg2pose_preprocess import preprocess_emg2pose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=1000)
    parser.add_argument("--window-ms", type=int, default=250)
    parser.add_argument("--stride-ms", type=int, default=125)
    return parser.parse_args()

def main():
    args = parse_args()

    # resolve data_root: CLI > env > ./data
    data_root = (
        Path(args.data_root)
        if args.data_root is not None
        else Path(os.environ.get("DATA_PATH", "./data"))
    )

    preprocess_emg2pose(
        data_root=data_root,
        target_fs=args.sampling_rate,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
    )

if __name__ == "__main__":
    main()
