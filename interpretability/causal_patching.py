#!/usr/bin/env python3
"""
Mechanistic Interpretability: Layer-Wise Causal Activation Patching.

Causal activation patching isolates where multimodal conflict arbitration occurs
by replacing intermediate hidden states of a corrupted forward pass with activations
from a clean visual forward pass.
"""

import torch
import torch.nn as nn
from evaluation.common import get_label_token_candidates, get_best_token_logit

class CleanActivationCatcher:
    """Hooks and captures hidden states across decoder layers during clean forward pass."""
    def __init__(self):
        self.activations = {}

    def get_hook(self, layer_idx):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Store detached activation on CPU or GPU in FP32
            self.activations[layer_idx] = hidden.detach().clone()
        return hook

class ActivationPatcher:
    """Hooks a specific layer during corrupted forward pass and overwrites hidden states."""
    def __init__(self, patch_tensor):
        self.patch_tensor = patch_tensor

    def hook(self, module, args, output):
        if isinstance(output, tuple):
            patched_hidden = self.patch_tensor.to(output[0].device, dtype=output[0].dtype)
            return (patched_hidden,) + output[1:]
        else:
            return self.patch_tensor.to(output.device, dtype=output.dtype)

def resolve_decoder_layers(model):
    """Dynamically locates the transformer decoder layers across model architectures."""
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
        lambda m: m.visual.blocks,
        lambda m: m.model.decoder.layers
    ]
    for cand in candidates:
        try:
            layers = cand(model)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except Exception:
            continue
    raise AttributeError("Could not automatically resolve decoder layers for model.")

def run_causal_patching_trial(
    model,
    processor,
    clean_inputs,
    corrupted_inputs,
    source_label,
    target_label,
    decoder_layers=None
):
    """
    Executes a single causal activation patching trial across all decoder layers.
    Returns:
        dict with layer indices and causal recovery scores.
    """
    if decoder_layers is None:
        decoder_layers = resolve_decoder_layers(model)

    num_layers = len(decoder_layers)
    src_token_ids = get_label_token_candidates(processor, source_label)
    tgt_token_ids = get_label_token_candidates(processor, target_label)

    # 1. Clean Forward Pass with Activation Hooks
    catcher = CleanActivationCatcher()
    clean_hooks = []
    for l_idx, layer in enumerate(decoder_layers):
        h = layer.register_forward_hook(catcher.get_hook(l_idx))
        clean_hooks.append(h)

    with torch.no_grad():
        clean_out = model(**clean_inputs)

    for h in clean_hooks:
        h.remove()

    clean_logits = clean_out.logits[0, -1, :].float()
    clean_src_logit = get_best_token_logit(clean_logits, src_token_ids)
    clean_tgt_logit = get_best_token_logit(clean_logits, tgt_token_ids)

    # 2. Corrupted Baseline Forward Pass
    with torch.no_grad():
        corrupt_out = model(**corrupted_inputs)

    corrupt_logits = corrupt_out.logits[0, -1, :].float()
    corrupt_src_logit = get_best_token_logit(corrupt_logits, src_token_ids)
    corrupt_tgt_logit = get_best_token_logit(corrupt_logits, tgt_token_ids)

    baseline_gap = clean_src_logit - corrupt_src_logit

    # 3. Layer-Wise Patching
    layer_recovery = []
    for l_idx in range(num_layers):
        patch_tensor = catcher.activations[l_idx]
        patcher = ActivationPatcher(patch_tensor)
        target_layer = decoder_layers[l_idx]
        patch_hook = target_layer.register_forward_hook(patcher.hook)

        with torch.no_grad():
            patched_out = model(**corrupted_inputs)

        patch_hook.remove()

        patched_logits = patched_out.logits[0, -1, :].float()
        patched_src_logit = get_best_token_logit(patched_logits, src_token_ids)
        patched_tgt_logit = get_best_token_logit(patched_logits, tgt_token_ids)

        if abs(baseline_gap) > 1e-4:
            recovery = (patched_src_logit - corrupt_src_logit) / baseline_gap
        else:
            recovery = 0.0

        layer_recovery.append({
            "layer": l_idx,
            "causal_recovery": float(recovery),
            "patched_source_logit": float(patched_src_logit),
            "patched_target_logit": float(patched_tgt_logit)
        })

    return {
        "num_layers": num_layers,
        "clean_source_logit": float(clean_src_logit),
        "corrupt_source_logit": float(corrupt_src_logit),
        "layers": layer_recovery
    }
