# utils/verify_extraction.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR

def verify():
    train_csv = os.path.join(DATASET_DIR, "train_keypoints.csv")
    df = pd.read_csv(train_csv)

    print("=" * 50)
    print("EXTRACTION VERIFICATION REPORT")
    print("=" * 50)
    print(f"Total training rows   : {len(df)}")
    print(f"Total feature columns : {len(df.columns) - 2}")  # minus label + filename
    print(f"Number of classes     : {df['label'].nunique()}")
    print(f"Missing values        : {df.isnull().sum().sum()}")

    print("\nTop 10 classes by image count:")
    print(df['label'].value_counts().head(10).to_string())

    print("\nBottom 10 classes by image count:")
    print(df['label'].value_counts().tail(10).to_string())

    # Check for any NaN in feature columns (indicates bad extraction)
    feature_cols = [c for c in df.columns if c not in ["label", "filename"]]
    nan_counts = df[feature_cols].isnull().sum()
    bad_cols = nan_counts[nan_counts > 0]
    if len(bad_cols) > 0:
        print(f"\nWARNING — columns with NaN values:\n{bad_cols}")
    else:
        print("\nAll feature columns are clean (no NaN values)")

    # Plot class distribution
    plt.figure(figsize=(14, 5))
    df['label'].value_counts().plot(kind='bar', color='steelblue', edgecolor='none')
    plt.title("Images per pose class (training set)")
    plt.xlabel("Pose class")
    plt.ylabel("Number of images")
    plt.xticks([])   # too many classes to label
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_DIR, "class_distribution.png"), dpi=150)
    print(f"\nClass distribution chart saved to dataset/class_distribution.png")
    plt.show()

if __name__ == "__main__":
    verify()