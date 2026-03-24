# extract_keypoints.py  (improved version)

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import json
from utils.label_mapper import get_label
from config import (
    DATASET_DIR, MODELS_DIR, LABELS_PATH,
    ANGLE_TRIPLETS, MP_DETECTION_CONFIDENCE, IMAGE_SIZE
)

mp_pose    = mp.solutions.pose

# ─── Angle calculation ────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ─── Feature extraction ───────────────────────────────────────────────────────
def extract_features(image_path, pose_detector):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Try original + horizontally flipped (doubles detection chances)
    for img in [image, cv2.flip(image, 1)]:
        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        results     = pose_detector.process(img_rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            coords     = []
            for lm in lms:
                coords.extend([lm.x, lm.y, lm.z])

            angles     = []
            for (a, b, c) in ANGLE_TRIPLETS:
                pt_a = (lms[a].x, lms[a].y)
                pt_b = (lms[b].x, lms[b].y)
                pt_c = (lms[c].x, lms[c].y)
                angles.append(calculate_angle(pt_a, pt_b, pt_c))

            visibility = [lm.visibility for lm in lms]

            return coords + angles + visibility

    return None


# ─── Column names ─────────────────────────────────────────────────────────────
def build_column_names():
    landmark_names = [
        "nose","left_eye_inner","left_eye","left_eye_outer",
        "right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_pinky","right_pinky",
        "left_index","right_index","left_thumb","right_thumb",
        "left_hip","right_hip","left_knee","right_knee",
        "left_ankle","right_ankle","left_heel","right_heel",
        "left_foot_index","right_foot_index"
    ]
    cols = []
    for n in landmark_names:
        cols += [f"{n}_x", f"{n}_y", f"{n}_z"]
    cols += [
        "left_elbow_angle","right_elbow_angle",
        "left_shoulder_angle","right_shoulder_angle",
        "left_hip_angle","right_hip_angle",
        "left_knee_angle","right_knee_angle",
        "left_ankle_angle","right_ankle_angle",
        "shoulder_width_angle","hip_width_angle",
    ]
    for n in landmark_names:
        cols.append(f"{n}_visibility")
    return cols


# ─── Process one split ────────────────────────────────────────────────────────
def process_split(split_name, class_names):
    split_dir   = os.path.join(DATASET_DIR, split_name)
    rows        = []
    skipped     = 0
    class_stats = {}

    print(f"\nProcessing '{split_name}' split...")

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.3
    ) as pose:

        for class_name in tqdm(class_names, desc=f"  Classes"):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            # ← ADD THIS LINE — converts folder name to clean label
            label = get_label(class_name)

            images = [
                f for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in {".jpg",".jpeg",".png",".webp"}
            ]

            extracted = 0
            for img_file in images:
                features = extract_features(os.path.join(class_dir, img_file), pose)
                if features is None:
                    skipped += 1
                    continue
                # ← CHANGED class_name to label here
                rows.append(features + [label, img_file])
                extracted += 1

            class_stats[class_name] = {"total": len(images), "extracted": extracted}
    # Print classes with poor extraction rates
    print(f"\n  Extracted {len(rows)} rows | Skipped {skipped} images")
    print(f"\n  Classes with extraction rate below 50%:")
    poor = {k: v for k, v in class_stats.items()
            if v["total"] > 0 and v["extracted"] / v["total"] < 0.5}
    if poor:
        for cls, s in sorted(poor.items(), key=lambda x: x[1]["extracted"]/x[1]["total"]):
            rate = s["extracted"] / s["total"] * 100
            print(f"    {cls:45s}  {s['extracted']:3d}/{s['total']:3d}  ({rate:.0f}%)")
    else:
        print("    None — all classes above 50%")

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_dir   = os.path.join(DATASET_DIR, "train")
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    # Drop classes with fewer than 8 training images — too few to learn from
    MIN_TRAIN_IMAGES = 8
    filtered = []
    dropped  = []
    for cls in class_names:
        cls_dir = os.path.join(train_dir, cls)
        n = len(os.listdir(cls_dir))
        if n >= MIN_TRAIN_IMAGES:
            filtered.append(cls)
        else:
            dropped.append(cls)

    print(f"Total classes found  : {len(class_names)}")
    print(f"Classes kept (>={MIN_TRAIN_IMAGES} imgs): {len(filtered)}")
    if dropped:
        print(f"Classes dropped      : {dropped}")

    class_names = filtered

    # Save final class list
    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Labels saved → {LABELS_PATH}  ({len(class_names)} classes)")

    columns = build_column_names() + ["label", "filename"]

    for split in ["train", "val", "test"]:
        rows    = process_split(split, class_names)
        df      = pd.DataFrame(rows, columns=columns)
        csv_out = os.path.join(DATASET_DIR, f"{split}_keypoints.csv")
        df.to_csv(csv_out, index=False)
        print(f"  Saved {csv_out}  ({len(df)} rows)")

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()