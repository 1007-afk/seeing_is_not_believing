#!/usr/bin/env python3
"""
Lightweight End-to-End Pilot Evaluation Runner.
Executes the full pipeline on a fast model (Qwen2-VL-2B-Instruct) across the 100-sample balanced dataset:
1. Verifies dataset and generated typographic injection images
2. Runs multimodal text-conflict and typographic prompt injection
3. Scores accuracy, text-gullibility adoption, and injection success
4. Saves scored metrics and summary tables in results/pilot/
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Pilot Evaluation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.csv")
    parser.add_argument("--injected_dir", type=str, default="data/injected_images")
    parser.add_argument("--output_dir", type=str, default="results/pilot")
    parser.add_argument("--max_samples", type=int, default=20, help="Number of samples to evaluate for quick test (default: 20)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1: PREPARING TYPOGRAPHIC INJECTION IMAGES")
    print("=" * 80)
    cmd_prep = [
        sys.executable, "data/generate_injected_images.py",
        "--manifest", args.manifest,
        "--output_dir", args.injected_dir
    ]
    subprocess.run(cmd_prep, check=True)

    print("\n" + "=" * 80)
    print(f"STEP 2: RUNNING EVALUATION SUITE WITH {args.model_id}")
    print("=" * 80)
    cmd_eval = [
        sys.executable, "evaluation/eval_qwen2_vl.py",
        "--model_id", args.model_id,
        "--manifest", args.manifest,
        "--injected_dir", args.injected_dir,
        "--output_dir", args.output_dir,
        "--seeds", "42"
    ]
    if args.max_samples:
        cmd_eval.extend(["--max_samples", str(args.max_samples)])
    if args.cpu:
        cmd_eval.append("--cpu")

    subprocess.run(cmd_eval, check=True)

    print("\n" + "=" * 80)
    print("STEP 3: SCORING EVALUATION RESULTS")
    print("=" * 80)
    cmd_score = [
        sys.executable, "analysis/score_results.py",
        "--results_dir", args.output_dir,
        "--output_dir", args.output_dir
    ]
    subprocess.run(cmd_score, check=True)

    print("\n" + "=" * 80)
    print(f"[+] PILOT EVALUATION COMPLETE! Scored metrics available in: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
