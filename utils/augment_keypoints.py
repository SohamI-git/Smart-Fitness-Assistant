# utils/augment_keypoints.py

import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR

AUGMENT_FACTOR = 5       # create 5 extra copies per original row
COORD_NOISE    = 0.01    # std dev for x, y, z noise (1% of normalized coords)
ANGLE_NOISE    = 2.0     # std dev for angle noise in degrees
VIS_NOISE      = 0.02    # std dev for visibility noise
RANDOM_SEED    = 42

def augment():
    csv_in  = os.path.join(DATASET_DIR, "train_keypoints.csv")
    csv_out = os.path.join(DATASET_DIR, "train_keypoints_aug.csv")

    print("Loading training keypoints...")
    df = pd.read_csv(csv_in)
    print(f"  Original rows : {len(df)}")

    feat_cols  = [c for c in df.columns if c not in ["label", "filename"]]
    coord_cols = [c for c in feat_cols if c.endswith(("_x", "_y", "_z"))]
    angle_cols = [c for c in feat_cols if c.endswith("_angle")]
    vis_cols   = [c for c in feat_cols if c.endswith("_visibility")]

    rng       = np.random.default_rng(RANDOM_SEED)
    aug_rows  = []

    for _, row in df.iterrows():
        for i in range(AUGMENT_FACTOR):
            new_row = row.copy()

            # Add small Gaussian noise to coordinates
            new_row[coord_cols] += rng.normal(0, COORD_NOISE, len(coord_cols))

            # Add small noise to joint angles
            new_row[angle_cols] += rng.normal(0, ANGLE_NOISE, len(angle_cols))

            # Clip angles to valid range [0, 180]
            new_row[angle_cols] = new_row[angle_cols].clip(0, 180)

            # Add tiny noise to visibility scores, clip to [0, 1]
            new_row[vis_cols] += rng.normal(0, VIS_NOISE, len(vis_cols))
            new_row[vis_cols]  = new_row[vis_cols].clip(0, 1)

            # Mark filename so we can identify augmented rows
            new_row["filename"] = f"aug_{i}_{row['filename']}"
            aug_rows.append(new_row)

    df_aug    = pd.DataFrame(aug_rows, columns=df.columns)
    df_final  = pd.concat([df, df_aug], ignore_index=True)

    # Shuffle so augmented rows are mixed with originals
    df_final  = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df_final.to_csv(csv_out, index=False)
    print(f"  Augmented rows: {len(df_aug)}")
    print(f"  Total rows    : {len(df_final)}")
    print(f"  Saved → {csv_out}")

    # Show class balance after augmentation
    print(f"\n  Rows per class (min / mean / max):")
    counts = df_final["label"].value_counts()
    print(f"    min  : {counts.min()}")
    print(f"    mean : {counts.mean():.0f}")
    print(f"    max  : {counts.max()}")

if __name__ == "__main__":
    augment()