# utils/fix_labels.py

import os
import pandas as pd
import sys

# ← this line fixes the import issue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_mapper import get_label          # ← relative import since we are inside utils/
from config import DATASET_DIR

for split in ["train", "val", "test"]:
    path = os.path.join(DATASET_DIR, f"{split}_keypoints.csv")
    if not os.path.exists(path):
        print(f"Skipping {split} — file not found")
        continue

    df = pd.read_csv(path)
    before = df["label"].nunique()
    df["label"] = df["label"].apply(get_label)
    after = df["label"].nunique()
    df.to_csv(path, index=False)
    print(f"{split}: {before} raw labels → {after} clean labels")
    print(f"  Sample: {df['label'].unique()[:5].tolist()}")