#!/usr/bin/env bash
# ==============================================================================
# Anonymous Reproduction Pipeline: Large-Scale Multimodal Bias & Conflict Suite
# ==============================================================================
set -e

MANIFEST="data/dataset_manifest.csv"
INJECTED_DIR="data/injected_images"
CONFIG="config/prompt_templates.json"
SEEDS="42 123 999"

echo "======================================================================"
echo "Step 1: Synthesizing Non-Destructive Typographic Banners"
echo "======================================================================"
python3 data/generate_injected_images.py --manifest "${MANIFEST}" --output_dir "${INJECTED_DIR}"

echo "======================================================================"
echo "Step 2: Executing Multimodal Conflict & Causal Patching Suite"
echo "======================================================================"

# 1. Qwen2-VL-7B
echo "[*] Running Qwen2-VL-7B-Instruct..."
python3 evaluation/eval_qwen2_vl.py \
    --model_id "Qwen/Qwen2-VL-7B-Instruct" \
    --manifest "${MANIFEST}" \
    --injected_dir "${INJECTED_DIR}" \
    --config "${CONFIG}" \
    --output_dir "results/qwen2_vl_7b" \
    --seeds ${SEEDS} \
    --run_patching \
    --max_patch_trials 25

# 2. Qwen2.5-VL-7B
echo "[*] Running Qwen2.5-VL-7B-Instruct..."
python3 evaluation/eval_qwen25_vl.py \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --manifest "${MANIFEST}" \
    --injected_dir "${INJECTED_DIR}" \
    --config "${CONFIG}" \
    --output_dir "results/qwen25_vl_7b" \
    --seeds ${SEEDS} \
    --run_patching \
    --max_patch_trials 25

# 3. LLaVA-1.5-7B
echo "[*] Running LLaVA-1.5-7B..."
python3 evaluation/eval_llava.py \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --manifest "${MANIFEST}" \
    --injected_dir "${INJECTED_DIR}" \
    --config "${CONFIG}" \
    --output_dir "results/llava_1.5_7b" \
    --seeds ${SEEDS} \
    --run_patching \
    --max_patch_trials 25

echo "======================================================================"
echo "Step 3: Scoring Results and Generating Figures"
echo "======================================================================"
for MODEL_DIR in "results/qwen2_vl_7b" "results/qwen25_vl_7b" "results/llava_1.5_7b"; do
    if [ -d "${MODEL_DIR}" ]; then
        echo "[*] Scoring ${MODEL_DIR}..."
        python3 analysis/score_results.py --results_dir "${MODEL_DIR}"
        python3 analysis/generate_visualizations.py --results_dir "${MODEL_DIR}"
    fi
done

echo "======================================================================"
echo "[+] REPRODUCTION PIPELINE COMPLETED SUCCESSFULLY!"
echo "======================================================================"
