#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""
Download all pretrained OpenTSLM checkpoints from HuggingFace and save them
as local .pt files in models/.

Usage:
    python download_pretrained_models.py [--device cpu] [--output_dir models]

The saved .pt files can be used directly with compute_mcspu.py:
    python compute_mcspu.py --checkpoint models/stage3_har_flamingo.pt --model_type flamingo ...
    python compute_mcspu.py --checkpoint models/stage3_har_sp.pt       --model_type sp ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "open_flamingo"))

import torch
from opentslm.model.llm.OpenTSLM import OpenTSLM

# ---------------------------------------------------------------------------
# Model catalogue
# Each entry: (hf_repo_id, local_filename, model_type)
# ---------------------------------------------------------------------------

MODELS = [
    # SP variants (confirmed available via demo scripts)
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


def download_and_save(repo_id: str, out_path: str, device: str) -> bool:
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return True
    try:
        model = OpenTSLM.load_pretrained(repo_id, device=device)
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
                        help="Download only these filenames, e.g. --only stage3_har_flamingo.pt")
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
    for repo_id, filename, _ in targets:
        out_path = os.path.join(args.output_dir, filename)
        print(f"--- {filename} ({repo_id})")
        ok = download_and_save(repo_id, out_path, device)
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
