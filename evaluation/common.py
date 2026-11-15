#!/usr/bin/env python3
"""
Shared utilities, prompt formatting, regex parsers, and evaluation metrics.
"""

import re
import json
import os
import torch
import pandas as pd
import numpy as np

TARGET_GROUPS = ["Black", "White", "Indian", "East Asian", "Middle Eastern"]

# Regular expressions for demographic category extraction
REGEX_PATTERNS = {
    "East Asian": re.compile(r"\b(east\s+asian|asian)\b", re.IGNORECASE),
    "Middle Eastern": re.compile(r"\b(middle\s+eastern|arab|persian)\b", re.IGNORECASE),
    "Black": re.compile(r"\b(black|african(\s+american)?)\b", re.IGNORECASE),
    "White": re.compile(r"\b(white|caucasian|european)\b", re.IGNORECASE),
    "Indian": re.compile(r"\b(indian|south\s+asian)\b", re.IGNORECASE),
    "Cannot determine": re.compile(r"\b(cannot\s+determine|can't\s+determine|unable\s+to\s+determine|insufficient|unclear)\b", re.IGNORECASE)
}

def load_prompt_templates(config_path="config/prompt_templates.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt template configuration missing at: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def parse_model_prediction(text_response):
    """
    Deterministically parses model response into one of the canonical demographic categories.
    Prioritizes explicit multi-word matches (e.g. 'East Asian' before 'Asian', 'Middle Eastern' before 'White').
    """
    if not text_response or not isinstance(text_response, str):
        return "Cannot determine"

    text_lower = text_response.lower()

    # Priority order for multi-word compounds
    check_order = ["Middle Eastern", "East Asian", "Cannot determine", "Black", "White", "Indian"]

    # First pass: check direct word boundary matches
    matches = []
    for cat in check_order:
        pattern = REGEX_PATTERNS[cat]
        match = pattern.search(text_lower)
        if match:
            matches.append((match.start(), cat))

    if matches:
        # Return the earliest mentioned category in the response
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    # Fallback to simple substring
    for cat in ["Middle Eastern", "East Asian", "Black", "White", "Indian"]:
        if cat.lower() in text_lower:
            return cat

    return "Cannot determine"

def get_label_token_candidates(processor, label_str):
    """
    Extracts token IDs corresponding to label candidates (handling space prefix variants).
    """
    candidates = [label_str, f" {label_str}"]
    if label_str == "East Asian":
        candidates.extend(["Asian", " Asian", "East", " East"])
    elif label_str == "Middle Eastern":
        candidates.extend(["Middle", " Middle", "Arab", " Arab"])

    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = []
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])
    return list(set(token_ids))

def get_best_token_logit(logits_1d, candidate_ids):
    """
    Retrieves the maximum logit across a set of candidate subword token IDs.
    """
    if not candidate_ids:
        return -100.0
    return max([logits_1d[tid].item() for tid in candidate_ids])
