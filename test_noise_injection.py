#!/usr/bin/env python3
"""Quick verification that noise injection works for all 4 dataset classes."""

import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "src/open_flamingo")

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def get_signal_flat(sample):
    """Extract flattened signal array from a sample."""
    ts = sample["time_series"]
    # ts is a list of arrays (one per channel/series)
    return np.concatenate([np.array(t).flatten() for t in ts])


def test_dataset(cls_name, DatasetClass, extra_kwargs=None):
    print(f"\n{'='*60}")
    print(f"Testing {cls_name}")
    print(f"{'='*60}")

    kwargs = {"split": "test", "EOS_TOKEN": "</s>"}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # 1) Load without noise
    DatasetClass.set_noise_mode(use_noise=False)
    ds_real = DatasetClass(**kwargs)
    sample_real = ds_real[0]
    sig_real = get_signal_flat(sample_real)
    print(f"  Loaded {len(ds_real)} test samples (real signal)")
    print(f"  Signal shape: {sig_real.shape}, mean={sig_real.mean():.4f}, std={sig_real.std():.4f}")

    # 2) Load with Gaussian noise
    DatasetClass.set_noise_mode(use_noise=True, noise_type="gaussian", noise_seed=67)
    ds_noise = DatasetClass(**kwargs)
    sample_noise = ds_noise[0]
    sig_noise = get_signal_flat(sample_noise)
    print(f"  Noise signal shape: {sig_noise.shape}, mean={sig_noise.mean():.4f}, std={sig_noise.std():.4f}")

    # 3) Check signals are different
    check("Signals differ", not np.allclose(sig_real, sig_noise, atol=1e-3))

    # 4) Check correlation is low
    if len(sig_real) == len(sig_noise) and sig_real.std() > 0 and sig_noise.std() > 0:
        corr = np.corrcoef(sig_real, sig_noise)[0, 1]
        check(f"Low correlation (|r|={abs(corr):.4f})", abs(corr) < 0.3)
    else:
        check("Same length", len(sig_real) == len(sig_noise),
              f"real={len(sig_real)} noise={len(sig_noise)}")

    # 5) Check text prompts preserved (should still mention original stats)
    text_real = " ".join(sample_real.get("time_series_text", []))
    text_noise = " ".join(sample_noise.get("time_series_text", []))
    check("Text prompts match (original stats preserved)", text_real == text_noise,
          f"\n    real:  {text_real[:100]}\n    noise: {text_noise[:100]}")

    # 6) Check different samples get different noise
    if len(ds_noise) > 1:
        sig_noise_1 = get_signal_flat(ds_noise[1])
        check("Different samples get different noise", not np.allclose(sig_noise, sig_noise_1, atol=1e-3))

    # 7) Test zero noise
    DatasetClass.set_noise_mode(use_noise=True, noise_type="zero", noise_seed=67)
    ds_zero = DatasetClass(**kwargs)
    sig_zero = get_signal_flat(ds_zero[0])
    check("Zero noise produces zeros", np.allclose(sig_zero, 0, atol=1e-6))

    # Reset
    DatasetClass.set_noise_mode(use_noise=False)


# ---- TSQA ----
from opentslm.time_series_datasets.TSQADataset import TSQADataset
test_dataset("TSQADataset", TSQADataset)

# ---- HAR-CoT ----
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
test_dataset("HARCoTQADataset", HARCoTQADataset)

# ---- Sleep-CoT ----
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
test_dataset("SleepEDFCoTQADataset", SleepEDFCoTQADataset)

# ---- ECG-CoT ----
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
test_dataset("ECGQACoTQADataset", ECGQACoTQADataset, extra_kwargs={"eval_only": True})

# ---- Summary ----
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} checks")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
