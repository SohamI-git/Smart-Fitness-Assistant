# utils/check_dataset_structure.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR

def check():
    yoga_dir = os.path.join(DATASET_DIR, "Yoga-82")

    if not os.path.exists(yoga_dir):
        print(f"dataset/Yoga-82/ folder does not exist yet.")
        print(f"Expected location: {yoga_dir}")
        return

    print(f"Found: {yoga_dir}")
    print(f"\nContents of dataset/Yoga-82/:")

    for root, dirs, files in os.walk(yoga_dir):
        depth = root.replace(yoga_dir, "").count(os.sep)
        if depth > 2:
            continue                      # don't go too deep
        indent  = "  " * depth
        relpath = os.path.relpath(root, yoga_dir)
        n_files = len(files)
        n_dirs  = len(dirs)
        print(f"{indent}{relpath}/  [{n_dirs} folders, {n_files} files]")

        if depth == 2 and n_files > 0:
            sample = files[:3]
            print(f"{'  ' * (depth+1)}sample files: {sample}")

if __name__ == "__main__":
    check()