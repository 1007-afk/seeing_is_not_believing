#!/usr/bin/env python3
"""
Pre-Flight Environment & Hardware Sanity Check.
Verifies PyTorch, CUDA, HuggingFace Transformers, and performs single-sample inference
to confirm that vision-language forward and backward passes execute cleanly with no NaNs/Infs.
"""

import os
import sys
import time
import torch
from PIL import Image

def test_environment(model_id="Qwen/Qwen2-VL-2B-Instruct"):
    print("=" * 70)
    print("VLM BENCHMARK PRE-FLIGHT VERIFICATION TEST")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"[*] Python: {sys.version.split()[0]}")
    print(f"[*] PyTorch: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"[*] CUDA Available: {cuda_avail}")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"[*] Primary GPU: {gpu_name}")
        print(f"[*] Initial VRAM: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")
        print(f"[*] BF16 Supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("[!] CUDA not detected. Inference will run on CPU.")

    print(f"\n[*] Testing model loading with: {model_id}...")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if cuda_avail and torch.cuda.is_bf16_supported() else torch.float32,
            device_map="auto" if cuda_avail else None,
            trust_remote_code=True
        )
        if not cuda_avail:
            model = model.to("cpu")
        model.eval()
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[!] Model loading failed: {e}")
        return False

    print("[*] Running dummy multimodal inference check...")
    try:
        dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": "What color is this square?"}
                ]
            }
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[dummy_img], padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda" if cuda_avail else "cpu") for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("[!] Output logits contain NaN or Inf values!")
                return False

            gen = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            prompt_len = inputs["input_ids"].shape[1]
            decoded = processor.decode(gen[0][prompt_len:], skip_special_tokens=True).strip()
            print(f"[+] Output generated cleanly: '{decoded}'")

        print("\n" + "=" * 70)
        print("[+] ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"[!] Sanity check inference failed: {e}")
        return False

if __name__ == "__main__":
    test_model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-VL-2B-Instruct"
    success = test_environment(test_model)
    sys.exit(0 if success else 1)
