#!/usr/bin/env python3
"""
Mechanistic Interpretability: Logit Lens Projection.

Directly projects the intermediate representations of each transformer layer through
the final layer normalization and language model head (lm_head) into vocabulary logits,
allowing real-time observation of when semantic concepts form, persist, or collapse.
"""

import torch
import torch.nn as nn
from interpretability.causal_patching import resolve_decoder_layers
from evaluation.common import get_label_token_candidates, get_best_token_logit

def resolve_lm_head_and_norm(model):
    """Locates the final lm_head and final normalization layer."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "language_model"):
        lm_head = getattr(model.language_model, "lm_head", None)

    # Resolve final norm
    norm = None
    for cand in [
        lambda m: m.model.language_model.norm,
        lambda m: m.language_model.model.norm,
        lambda m: m.model.norm,
        lambda m: m.language_model.norm,
        lambda m: m.norm
    ]:
        try:
            n = cand(model)
            if n is not None:
                norm = n
                break
        except Exception:
            continue

    return lm_head, norm

def run_logit_lens(model, inputs, candidate_labels, processor, decoder_layers=None):
    """
    Computes Logit Lens scores across all decoder layers.
    Returns:
        dict mapping layer index to top predicted label and candidate logit values.
    """
    if decoder_layers is None:
        decoder_layers = resolve_decoder_layers(model)

    lm_head, final_norm = resolve_lm_head_and_norm(model)
    if lm_head is None:
        raise AttributeError("Could not resolve lm_head for Logit Lens projection.")

    layer_activations = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_activations[idx] = hidden[:, -1, :].detach().clone()
        return hook_fn

    for l_idx, layer in enumerate(decoder_layers):
        h = layer.register_forward_hook(make_hook(l_idx))
        hooks.append(h)

    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    results = []
    for l_idx in range(len(decoder_layers)):
        hidden = layer_activations[l_idx]
        if final_norm is not None:
            hidden = final_norm(hidden)

        logits = lm_head(hidden).squeeze(0).float()

        layer_info = {"layer": l_idx}
        for lbl in candidate_labels:
            c_ids = get_label_token_candidates(processor, lbl)
            layer_info[lbl] = float(get_best_token_logit(logits, c_ids))

        results.append(layer_info)

    return results
