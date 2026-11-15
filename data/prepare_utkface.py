#!/usr/bin/env python3
"""
Prepare UTKFace dataset manifest for multimodal evaluation.
UTKFace filenames follow the format: [age]_[gender]_[race]_[date&time].jpg
Race labels:
  0: White
  1: Black
  2: East Asian (Asian)
  3: Indian
  4: Others (Excluded or Middle Eastern)
"""

import os
import argparse
import glob
import pandas as pd
from PIL import Image

RACE_MAP = {
    "0": "White",
    "1": "Black",
    "2": "East Asian",
    "3": "Indian"
}

GENDER_MAP = {
    "0": "Male",
    "1": "Female"
}

def main():
    parser = argparse.ArgumentParser(description="Prepare UTKFace manifest")
    parser.add_argument("--utk_dir", type=str, default="data/utkface", help="Root directory containing UTKFace images")
    parser.add_argument("--output_manifest", type=str, default="data/utkface_manifest.csv", help="Output path for manifest CSV")
    args = parser.parse_args()

    if not os.path.exists(args.utk_dir):
        print(f"[!] UTKFace directory '{args.utk_dir}' does not exist.")
        print("    Please download UTKFace and place unzipped files in data/utkface/")
        return

    image_paths = glob.glob(os.path.join(args.utk_dir, "*.jpg")) + glob.glob(os.path.join(args.utk_dir, "*.png"))
    print(f"[*] Found {len(image_paths)} images in {args.utk_dir}")

    rows = []
    for idx, p in enumerate(image_paths):
        fname = os.path.basename(p)
        parts = fname.split("_")
        if len(parts) < 4:
            continue

        age_str, gender_str, race_str = parts[0], parts[1], parts[2]
        if race_str not in RACE_MAP:
            continue

        race_label = RACE_MAP[race_str]
        gender_label = GENDER_MAP.get(gender_str, "unknown")

        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            continue

        rows.append({
            "image_id": f"utk_{idx:06d}",
            "file_path": p,
            "dataset": "UTKFace",
            "original_label": race_label,
            "analysis_group": race_label,
            "gender": gender_label,
            "age": age_str,
            "image_width": w,
            "image_height": h
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_manifest, index=False)
    print(f"[+] Saved {len(df)} UTKFace items to {args.output_manifest}")

if __name__ == "__main__":
    main()
