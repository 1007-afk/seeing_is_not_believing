#!/usr/bin/env python3
"""
Prepare Balanced Faces in the Wild (BFW) dataset manifest for multimodal evaluation.
BFW structures demographic folders as:
  bfw/data/bfw-cropped-face/[ethnicity]_[gender]/[person_id]/[image_name]
Ethnicity mappings:
  - asian -> East Asian
  - black -> Black
  - indian -> Indian
  - white -> White
"""

import os
import argparse
import glob
import pandas as pd
from PIL import Image

ETHNICITY_MAP = {
    "asian": "East Asian",
    "black": "Black",
    "indian": "Indian",
    "white": "White"
}

def main():
    parser = argparse.ArgumentParser(description="Prepare BFW manifest")
    parser.add_argument("--bfw_dir", type=str, default="data/bfw", help="Root directory containing BFW images")
    parser.add_argument("--output_manifest", type=str, default="data/bfw_manifest.csv", help="Output path for manifest CSV")
    args = parser.parse_args()

    if not os.path.exists(args.bfw_dir):
        print(f"[!] BFW directory '{args.bfw_dir}' does not exist.")
        print("    Please download BFW from https://github.com/visionjo/facerec-bias-bfw and place in data/bfw/")
        return

    image_paths = glob.glob(os.path.join(args.bfw_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(args.bfw_dir, "**", "*.png"), recursive=True)

    print(f"[*] Found {len(image_paths)} images in {args.bfw_dir}")

    rows = []
    for idx, p in enumerate(image_paths):
        # Extract demographic tokens from path
        parts = p.split(os.sep)
        group_label = None
        gender_label = None

        for part in parts:
            part_lower = part.lower()
            for k, v in ETHNICITY_MAP.items():
                if part_lower.startswith(k):
                    group_label = v
                    if "female" in part_lower:
                        gender_label = "Female"
                    elif "male" in part_lower:
                        gender_label = "Male"
                    break
            if group_label:
                break

        if not group_label:
            continue

        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            continue

        rows.append({
            "image_id": f"bfw_{idx:06d}",
            "file_path": p,
            "dataset": "BFW",
            "original_label": group_label,
            "analysis_group": group_label,
            "gender": gender_label or "unknown",
            "image_width": w,
            "image_height": h
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_manifest, index=False)
    print(f"[+] Saved {len(df)} BFW items to {args.output_manifest}")

if __name__ == "__main__":
    main()
