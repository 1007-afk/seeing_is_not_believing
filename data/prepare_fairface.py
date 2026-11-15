#!/usr/bin/env python3
"""
Download and prepare the FairFace validation split (0.25 margin) for large-scale evaluation.
Source: HuggingFaceM4/FairFace (split='validation', margin='0.25')
"""

import os
import argparse
import pandas as pd
from PIL import Image

TARGET_GROUPS = ["Black", "White", "Indian", "East Asian", "Middle Eastern"]

def main():
    parser = argparse.ArgumentParser(description="Download and prepare FairFace validation set")
    parser.add_argument("--output_dir", type=str, default="data/fairface", help="Directory to save images and manifest")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional limit on number of samples (default: all)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    manifest_csv = os.path.join(args.output_dir, "fairface_manifest.csv")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets via 'pip install datasets'")

    print("[*] Streaming FairFace validation split from HuggingFace...")
    ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="validation", streaming=True)

    rows = []
    race_names = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']

    count = 0
    for idx, item in enumerate(ds):
        race_idx = item.get("race")
        race_label = race_names[race_idx] if isinstance(race_idx, int) and race_idx < len(race_names) else str(race_idx)

        # Standardize to 5 primary evaluation categories
        if race_label not in TARGET_GROUPS:
            continue

        img_id = f"ff_{idx:06d}"
        img_filename = f"{img_id}.png"
        img_rel_path = os.path.join(args.output_dir, "images", img_filename)

        img = item["image"]
        img.save(img_rel_path, format="PNG")

        rows.append({
            "image_id": img_id,
            "file_path": img_rel_path,
            "dataset": "FairFace",
            "original_label": race_label,
            "analysis_group": race_label,
            "gender": item.get("gender", "unknown"),
            "age": item.get("age", "unknown"),
            "image_width": img.width,
            "image_height": img.height
        })

        count += 1
        if count % 500 == 0:
            print(f"  [+] Saved {count} FairFace images...")

        if args.sample_size and count >= args.sample_size:
            break

    df = pd.DataFrame(rows)
    df.to_csv(manifest_csv, index=False)
    print(f"\n[+] Successfully saved {len(df)} images to {args.output_dir}/images and manifest to {manifest_csv}")

if __name__ == "__main__":
    main()
