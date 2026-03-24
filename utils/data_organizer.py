# utils/data_organizer.py

import os
import shutil
import random
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR

# ─── Config ──────────────────────────────────────────────────────────────────
SOURCE_DIR  = os.path.join(DATASET_DIR, "Yoga-82", "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

TRAIN_RATIO = 0.70   # 70% training
VAL_RATIO   = 0.15   # 15% validation
TEST_RATIO  = 0.15   # 15% testing

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_SEED = 42

# ─── Main ─────────────────────────────────────────────────────────────────────
def organize_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Please download Yoga-82 and place it in dataset/Yoga-82/Classes/")
        return

    # Get all class folders
    class_names = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])
    print(f"Found {len(class_names)} pose classes")

    total_copied = 0
    class_counts = {}

    for class_name in tqdm(class_names, desc="Organizing classes"):
        src_class_dir = os.path.join(SOURCE_DIR, class_name)

        # Collect all valid images in this class
        images = [
            f for f in os.listdir(src_class_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]

        if len(images) == 0:
            print(f"  WARNING: No images found for class '{class_name}'")
            continue

        # Shuffle for randomness
        random.seed(RANDOM_SEED)
        random.shuffle(images)

        # Split indices
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        class_counts[class_name] = {k: len(v) for k, v in splits.items()}

        # Copy images into split folders
        for split_name, split_images in splits.items():
            dest_dir = os.path.join(DATASET_DIR, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for img_file in split_images:
                src  = os.path.join(src_class_dir, img_file)
                dest = os.path.join(dest_dir, img_file)
                shutil.copy2(src, dest)
                total_copied += 1

    # ─── Summary report ──────────────────────────────────────────────────────
    print(f"\nDataset organized successfully!")
    print(f"Total images copied: {total_copied}")
    print(f"\nSplit breakdown:")
    total_train = sum(v["train"] for v in class_counts.values())
    total_val   = sum(v["val"]   for v in class_counts.values())
    total_test  = sum(v["test"]  for v in class_counts.values())
    print(f"  Train : {total_train} images")
    print(f"  Val   : {total_val}   images")
    print(f"  Test  : {total_test}  images")

    # Show classes with very few images (potential problem classes)
    print(f"\nClasses with fewer than 10 training images:")
    for cls, counts in class_counts.items():
        if counts["train"] < 10:
            print(f"  {cls}: {counts['train']} train images")

    return class_counts


if __name__ == "__main__":
    organize_dataset()