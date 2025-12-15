import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import numpy as np
import cv2

from notebook.utils import setup_sam_3d_body
from sam_3d_body.metadata.mhr70 import pose_info


def joint_angle(p_prev, p_mid, p_next, eps=1e-8):
    v1 = p_prev - p_mid
    v2 = p_next - p_mid

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return np.nan

    cos_theta = np.dot(v1, v2) / (n1 * n2 + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def chain_to_indices(chain_names, name_to_idx):
    return [name_to_idx[name] for name in chain_names]


def compute_finger_angles(joints_3d, chain_names, name_to_idx):
    idx = chain_to_indices(chain_names, name_to_idx)
    pts = [joints_3d[i] for i in idx]
    angle1 = joint_angle(pts[0], pts[1], pts[2])  # proximal
    angle2 = joint_angle(pts[1], pts[2], pts[3])  # distal
    return angle1, angle2


def main():
    parser = argparse.ArgumentParser(
        description="Hand joint angle labeling for RIGHT hand using SAM 3D Body on video"
    )
    parser.add_argument("--video", type=str, default="hand.mp4", help="Path to input video")
    parser.add_argument("--out_csv", type=str, default="hand_angles.csv", help="Output CSV path")
    parser.add_argument("--max_frames", type=int, default=-1, help="Process at most N frames (-1 = all)")
    parser.add_argument("--every_n", type=int, default=1, help="Process every Nth frame (1 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    # --- 1) Setup estimator once ---
    print("Setting up SAM 3D Body estimator")
    estimator = setup_sam_3d_body(
        hf_repo_id="facebook/sam-3d-body-dinov3",
        detector_name=None,
        segmentor_name=None,
        fov_name=None,
        device="cpu",
    )

    # --- 2) Metadata name->idx ---
    mhr_kp_index = {int(idx): info["name"] for idx, info in pose_info["keypoint_info"].items()}
    name_to_idx = {name: idx for idx, name in mhr_kp_index.items()}

    RIGHT_HAND_CHAINS = {
        "thumb":  ["right_thumb_third_joint", "right_thumb2", "right_thumb3", "right_thumb4"],
        "index":  ["right_forefinger_third_joint", "right_forefinger2", "right_forefinger3", "right_forefinger4"],
        "middle": ["right_middle_finger_third_joint", "right_middle_finger2", "right_middle_finger3", "right_middle_finger4"],
        "ring":   ["right_ring_finger_third_joint", "right_ring_finger2", "right_ring_finger3", "right_ring_finger4"],
        "pinky":  ["right_pinky_finger_third_joint", "right_pinky_finger2", "right_pinky_finger3", "right_pinky_finger4"],
    }

    # sanity
    for finger, chain in RIGHT_HAND_CHAINS.items():
        for name in chain:
            if name not in name_to_idx:
                raise KeyError(f"Keypoint name not found in metadata: '{name}' (finger={finger})")

    # --- 3) Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback
    print(f"Video opened. FPS={fps:.3f}")

    # --- 4) CSV header ---
    # columns: frame/time + proximal/distal for each finger
    fieldnames = ["frame_idx", "time_sec"]
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        fieldnames += [
            f"{finger}_proximal_deg",
            f"{finger}_distal_deg",
        ]

    # --- 5) Process frames ---
    rows_written = 0
    frame_idx = -1

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1

            if args.every_n > 1 and (frame_idx % args.every_n != 0):
                continue
            if args.max_frames > 0 and rows_written >= args.max_frames:
                break

            time_sec = frame_idx / fps

            # estimator.process_one_image expects RGB numpy array in many setups
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ---- run SAM3D on this frame ----
            outputs = estimator.process_one_image(frame_rgb)

            row = {"frame_idx": frame_idx, "time_sec": time_sec}

            if not outputs:
                # no person detected -> NaNs
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    row[f"{finger}_proximal_deg"] = np.nan
                    row[f"{finger}_distal_deg"] = np.nan
                writer.writerow(row)
                rows_written += 1
                continue

            person = outputs[0]
            joints_3d = np.asarray(person["pred_keypoints_3d"])  # (70,3)

            # compute angles
            for finger, chain in RIGHT_HAND_CHAINS.items():
                a1, a2 = compute_finger_angles(joints_3d, chain, name_to_idx)
                row[f"{finger}_proximal_deg"] = a1
                row[f"{finger}_distal_deg"] = a2

            writer.writerow(row)
            rows_written += 1

            if rows_written % 25 == 0:
                print(f"Processed {rows_written} frames (last frame_idx={frame_idx})")

    cap.release()
    print(f"Done. Wrote {rows_written} rows to: {args.out_csv}")


if __name__ == "__main__":
    main()
