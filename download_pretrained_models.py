#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""
Download all pretrained OpenTSLM checkpoints from HuggingFace and save them
as local .pt files in models/.

Run from the repo root:

    python scripts/download_pretrained_models.py [--device cpu] [--output_dir models]
    python scripts/download_pretrained_models.py --only stage3_har_sp.pt stage4_sleep_sp.pt

The saved .pt files work directly with compute_mcspu.py and
opentslm_uncertainty_test.py.
"""

import argparse
import os
import sys

# src/ lives one level above this script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "open_flamingo"))

import torch
from opentslm.model.llm.OpenTSLM import OpenTSLM

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

MODELS = [
    # SP variants
    ("OpenTSLM/llama-3.2-1b-tsqa-sp",  "stage1_tsqa_sp.pt",      "sp"),
    ("OpenTSLM/llama-3.2-1b-m4-sp",    "stage2_m4_sp.pt",        "sp"),
    ("OpenTSLM/llama-3.2-1b-har-sp",   "stage3_har_sp.pt",       "sp"),
    ("OpenTSLM/llama-3.2-1b-sleep-sp", "stage4_sleep_sp.pt",     "sp"),
    ("OpenTSLM/llama-3.2-1b-ecg-sp",   "stage5_ecg_sp.pt",       "sp"),

    # Flamingo variants
    ("OpenTSLM/llama-3.2-1b-har-flamingo",   "stage3_har_flamingo.pt",   "flamingo"),
    ("OpenTSLM/llama-3.2-1b-sleep-flamingo", "stage4_sleep_flamingo.pt", "flamingo"),
    ("OpenTSLM/llama-3.2-1b-ecg-flamingo",   "stage5_ecg_flamingo.pt",   "flamingo"),
]


def _inspect_raw_checkpoint(path: str):
    """Print gate-related keys and values from a raw HuggingFace checkpoint."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    top_keys = list(raw.keys())
    print(f"  [raw ckpt] top-level keys: {top_keys}")

    # Flatten whichever sub-dict holds the actual state
    state = raw
    for k in ("llm", "model_state", "state_dict", "model"):
        if k in raw and isinstance(raw[k], dict):
            state = raw[k]
            print(f"  [raw ckpt] using sub-dict '{k}' ({len(state)} keys)")
            break

    gate_keys = [k for k in state if "gate" in k.lower()]
    if not gate_keys:
        print("  [raw ckpt] ⚠  NO gate keys found — gates were never saved in this checkpoint")
    else:
        print(f"  [raw ckpt] {len(gate_keys)} gate keys found:")
        for k in gate_keys[:12]:
            v = state[k]
            val = v.item() if v.numel() == 1 else f"shape={tuple(v.shape)}"
            print(f"             {k}: {val}")
        if len(gate_keys) > 12:
            print(f"             ... and {len(gate_keys) - 12} more")


def _check_flamingo_gates_loaded(model) -> None:
    """Print tanh(gate) values after loading to confirm they are non-zero."""
    try:
        lang_enc = model.model.lang_encoder
        layers = getattr(lang_enc, "gated_cross_attn_layers", None)
        if layers is None:
            print("  [gate check] gated_cross_attn_layers not found")
            return
        vals = [float(torch.tanh(l.attn_gate).item())
                for l in layers if l is not None]
        if not vals:
            print("  [gate check] no active layers")
            return
        import numpy as np
        max_abs = float(np.max(np.abs(vals)))
        print(f"  [gate check] {len(vals)} layers  "
              f"tanh(attn_gate) mean={float(np.mean(vals)):.6f}  max|·|={max_abs:.6f}")
        if max_abs < 1e-4:
            print("  [gate check] ⚠  All gates are zero — "
                  "model will ignore the signal (MCSPU will be ~0)")
        else:
            print("  [gate check] ✓  Gates are non-zero — signal path is active")
    except Exception as e:
        print(f"  [gate check] could not read gates: {e}")


def download_and_save(repo_id: str, out_path: str, device: str, model_type: str) -> bool:
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return True
    try:
        from huggingface_hub import hf_hub_download
        raw_path = hf_hub_download(repo_id=repo_id, filename="model_checkpoint.pt")
        print(f"  [inspect raw HuggingFace checkpoint]")
        _inspect_raw_checkpoint(raw_path)

        model = OpenTSLM.load_pretrained(repo_id, device=device)

        if model_type == "flamingo":
            _check_flamingo_gates_loaded(model)

        model.store_to_file(out_path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  [FAILED] {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pretrained OpenTSLM models")
    parser.add_argument("--device", default=None,
                        help="Device: cuda / cpu (default: auto-detect)")
    parser.add_argument("--output_dir", default="models",
                        help="Directory to save .pt files (default: models/)")
    parser.add_argument("--only", nargs="+", metavar="FILENAME",
                        help="Download only these filenames, e.g. --only stage3_har_sp.pt")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    targets = MODELS
    if args.only:
        targets = [m for m in MODELS if m[1] in args.only]
        if not targets:
            print(f"No models matched --only {args.only}")
            sys.exit(1)

    print(f"Saving to: {os.path.abspath(args.output_dir)}")
    print(f"Device:    {device}")
    print(f"Models:    {len(targets)}\n")

    results = []
    for repo_id, filename, model_type in targets:
        out_path = os.path.join(args.output_dir, filename)
        print(f"--- {filename} ({repo_id})")
        ok = download_and_save(repo_id, out_path, device, model_type)
        results.append((filename, ok))
        print()

    print("=" * 50)
    ok_count = sum(1 for _, ok in results if ok)
    print(f"Done: {ok_count}/{len(results)} succeeded")
    for filename, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {filename}")


if __name__ == "__main__":
    main()
