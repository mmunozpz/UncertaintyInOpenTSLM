# OpenTSLM-Uncertainty

Signal uncertainty quantification for [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) checkpoints, built on **Monte Carlo Signal Perturbation Uncertainty (MCSPU)**.

MCSPU measures how much a model's output distribution shifts when the input time-series is perturbed.  
Three perturbation modes are currently supported: gaussian noise (`gaussian`), missing timepoints (`missing_zeros`), and missing channels (`missing_channels`).

---

## Quick start: how to test a new checkpoint

The main entry point is `opentslm_uncertainty_test.py`. It runs a full 4-sigma gaussian noise MCSPU sweep on your model and decides if it is production-ready (checking if it truly relies on the signal).

```bash
# 1. Install dependencies (can be done using uv also)
pip install -r requirements.txt

# 2. Run the uncertainty test on your checkpoint
python opentslm_uncertainty_test.py \
    --checkpoint         models/<your_model>.pt \
    --dataset            <tsqa|har|sleep|ecg_qa> \
    --model_type         <sp|flamingo> \
    --perturbation_type  <gaussian|missing_zeros|missing_channels> \ # (default is gaussian)
    --llm_id             meta-llama/Llama-3.2-1B \
    --out_dir            plots \
    --n_noise            50 \
    --max_samples        200 \
    --device             cuda \
    --class_batch        16    # (adapt to your GPU capabilities)

```

> Use at least `--n_noise 50` and `--max_samples 200` (default). These values provide sufficient statistical power (Cohen's $d \ge 0.3$ detectable at $> 95\%$ power). See `UNCERTAINTY_TEST_GUIDE.txt` for the full justification.

Exit code `0` = PRODUCTION READY, `1` = NOT PRODUCTION READY.  
All modes will print a summary table in the terminal and save plots to `--out_dir`

---

## What is MCSPU?

**Monte Carlo Signal Perturbation Uncertainty** is a per-sample score defined as:

```
U_signal(x) = (1/N) Σᵢ  KL( p_clean ‖ p_perturbed_i )
```

where:

- `p_clean` = model output distribution on the real signal
- `p_perturbed_i` = model output distribution on the perturbed signal (noise or missing data)
- `N` = number of perturbation draws (default 50)

The MCSPU scorer (`src/opentslm/uncertainty/mcspu.py`) operates by replacing the real signal with a perturbed copy and comparing the resulting output distributions via KL divergence. The text description in the prompt is intentionally left unchanged so only signal content is perturbed.

**High MCSPU** → model output distribution shifts when the signal is perturbed → the model is using the signal.  
**Low MCSPU** → model output distribution barely changes → the model is ignoring the signal and answering from text/prior only.

### Gaussian noise mode

The perturbation is additive: signal + εᵢ, εᵢ ~ N(0, σ²). The test sweeps σ ∈ {0.1, 0.5, 1.0, 2.0} and checks that uncertainty increases with noise magnitude. This is the production gate, we can use it to confirm our models are ready for deployment.

Click [here](https://docs.google.com/presentation/d/1ry3dPZKT27JIxMl1WJfr58Jk8dr9dkhqISxPgbPBass/edit?usp=sharing) to understand the background and the experiments that led to this test.

<p align="center">
  <img src="assets/mcspu_vs_noise.png" width="500">
</p>

### Missing data modes

Two missing-data modes are available. Both always exit 0, this is just exploratory and has no thresholds, it produces plots for encoder design guidance.

**`missing_zeros`**: a random fraction of timepoints per channel is set to 0.0. Each of the N draws uses a different random mask. Sweeps fractions ∈ {0.1, 0.25, 0.5, 0.75, 1.0}. Tests how quickly the model degrades as signal data goes missing over time.

**`missing_channels`**: entire channels are blacked out completely. Some presets have been defined for har and ecg datasets by default, but custom configs via `--drop_channels` can also be used. This mode can help us show which input channels the encoder really relies on and are most important.

---

## Tests for gaussian mode

Every checkpoint must pass all tests to be production-ready. The first four are just checking that calculations makes sense, a failure there means broken computation not a bad model, and the last four are the real tests that must be passed in order to be production ready, that means that the model is truly relying on the signal.

#### Sanity tests

A failure here means broken computation, not a bad model:

- `no_nan_inf_scores`: 0 non-finite MCSPU scores
- `kl_nonneg` : min KL ≥ −1 × 10⁻⁶ nats
- `score_consistency`: max \|score − mean(KL)\| ≤ 1 × 10⁻⁵ nats
- `probs_normalized` : max \|Σ probs − 1\| ≤ 1 × 10⁻³

**Signal sensitivity tests:**

- `sensitivity_magnitude`: mean MCSPU(σ=2.0) > 0.05 nats (sp) and 0.008 nats (flamingo)
- `sensitivity_range`: MCSPU(σ=2.0) − MCSPU(σ=0.1) > 0.02 nats (sp) and 0.008 nats (flamingo)
- `statistical_significance`: Mann-Whitney U p < 0.05
- `effect_size`: Cohen's d > 0.30

Thresholds were derived from the HuggingFace pretrained checkpoints. Flamingo thresholds are lower because the cross-attention gates remain near-zero early in training, structurally suppressing absolute KL by ~10-20×. See `UNCERTAINTY_TEST_GUIDE.txt` for full justification of both sets.

The missing-data modes have no thresholds as results are exploratory only to help us in encoder design decisions.

---

## How the test was derived

> **You do not need to run any of this.** The pipeline below is how the thresholds and the production gate were originally established against the upstream pretrained checkpoints. It is documented here for reproducibility and transparency.

The derivation ran in four steps:

1. **Download** the official OpenTSLM checkpoints from HuggingFace via `scripts/download_pretrained_models.py`.
2. **Sweep** σ ∈ {0.1, 0.5, 1.0, 2.0} across all datasets using `scripts/run_tsqa_har_sleep_gpu.sh` (TSQA / HAR / Sleep) and `scripts/run_ecg_gpu_batched.sh` (ECG-QA). Each run writes a JSONL file of per-sample KL scores via `scripts/compute_mcspu.py`.
3. **Visualise** with `scripts/plot_mcspu.py` (reads all JSONL files, outputs figures including the `mcspu_vs_sigma.png` shown above).
4. **Derive thresholds** from the resulting distributions. See `UNCERTAINTY_TEST_GUIDE.txt` for the full statistical justification.

---

## Contents

```
OpenTSLM-Uncertainty/
├── opentslm_uncertainty_test.py   # MAIN SCRIPT — run this on any new checkpoint
├── UNCERTAINTY_TEST_GUIDE.txt     # Full test documentation and threshold derivations
├── requirements.txt               # Python dependencies
├── pyproject.toml
├── scripts/
│   ├── compute_mcspu.py           # Low-level MCSPU scorer (one sigma at a time)
│   ├── plot_mcspu.py              # Plotting tool for pre-computed JSONL results
│   ├── download_pretrained_models.py  # Download official OpenTSLM checkpoints
│   ├── run_tsqa_har_sleep_gpu.sh  # GPU sigma sweep for TSQA / HAR / Sleep EDF
│   └── run_ecg_gpu_batched.sh     # GPU sigma sweep for ECG-QA (OOM-safe batching)
└── src/
    ├── opentslm/                  # OpenTSLM library (model, datasets, uncertainty)
    └── data/har_cot/              # HAR chain-of-thought CSV splits
```

---

## References

- OpenTSLM: [github.com/StanfordBDHG/OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM)
- MCSPU methodology: see `UNCERTAINTY_TEST_GUIDE.txt` in this repo
