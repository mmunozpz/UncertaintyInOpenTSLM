# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenTSLM is a research framework for integrating time series as a native modality into LLMs, enabling natural-language prompting over multivariate medical time-series data (ECG, EEG, accelerometer, etc.). It implements multi-task curriculum learning across 5 progressive stages.

Two model architectures exist:
- **OpenTSLMSP** — frozen LLM + trainable encoder/projector (soft prompt injection)
- **OpenTSLMFlamingo** — gated cross-attention integration with `SignalContributionTracker` for interpretability

## Environment Setup

Requires Python 3.12. Preferred tool is `uv`:

```bash
uv sync --all-groups
source .venv/bin/activate
```

Alternative:
```bash
pip install -r requirements.txt
```

## Common Commands

**Run curriculum training:**
```bash
python curriculum_learning.py --model OpenTSLMFlamingo
python curriculum_learning.py --model OpenTSLMSP --device cuda --batch_size 4 --gradient_checkpointing
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage1_mcq stage2_captioning
```

**Run demo inference scripts:**
```bash
python demo/huggingface/01_test_hf_tsqa.py       # Stage 1: MCQ
python demo/huggingface/05_test_hf_ecg_qa_cot.py # Stage 5: ECG QA CoT
```

**Interpretability analysis:**
```bash
python test_noise_injection.py                                        # Verify noise injection across all 4 dataset classes
python measure_signal_contribution.py --checkpoint <path/to/best_model.pt>  # Measure per-layer ECG signal contribution (--max_samples, --device optional)
```

**Tests** (any file in `test/` can be run directly):
```bash
python test/smoke_test.py
python test/test_curriculum_trainer.py
python test/test_inference.py
python test/ecg_qa_cot_test.py
```

**Lint (ruff, line-length 120, target py312):**
```bash
ruff check src/ test/
ruff format src/ test/
```

**REUSE compliance** (enforced by CI):
```bash
reuse lint
```

**Baseline evaluation** (requires `eval` dependency group — `uv sync --group eval`):
```bash
python evaluation/baseline/evaluate_all.py
```

## Architecture

### Data Flow
```
Raw Time Series → Encoder (TransformerCNNEncoder) → Projector (MLPProjector)
                                                     ↓
                                         Soft prompt injection (SP)
                                         or cross-attention (Flamingo)
                                                     ↓
                                             LLM → text output
```

### Core Module Relationships

**`src/opentslm/`**

- `model/llm/OpenTSLM.py` — factory; detects model type from HuggingFace repo ID suffix (`-sp` vs `-flamingo`) and calls the right constructor
- `model/llm/OpenTSLMSP.py` — SP variant; encoder + projector outputs prepended as soft tokens to the LLM input
- `model/llm/OpenTSLMFlamingo.py` — Flamingo variant; registers `SignalContributionTracker` forward hooks on cross-attention layers to quantify per-layer ECG signal contribution
- `model/encoder/TransformerCNNEncoder.py` — primary encoder: CNN patches (kernel=stride=4) → positional embeddings → 6-layer Transformer
- `model/projector/MLPProjector.py` — projects encoder output to LLM embedding dim
- `time_series_datasets/noise_mixin.py` — `NoiseInjectionMixin` used by all dataset classes; supports `gaussian`, `shuffle`, `zero`, `uniform` noise modes plus `_strip_stats` to remove mean/std from text prompts
- `model_config.py` — central hyperparameters: `PATCH_SIZE=4`, `EMBED_DIM=128`, `LR_ENCODER=2e-4`, `LR_PROJECTOR=1e-4`, etc.

### Curriculum Stages (datasets)

Each stage's dataset inherits from both `QADataset` and `NoiseInjectionMixin`:

| Stage | Dataset class | Task |
|-------|--------------|------|
| 1 | `TSQADataset` | MCQ over time series |
| 2 | `M4QADataset` | Captioning |
| 3 | `HARCoTQADataset` | HAR chain-of-thought |
| 4 | `SleepEDFCoTQADataset` | Sleep staging CoT |
| 5 | `ECGQACoTQADataset` | ECG QA CoT |

`curriculum_learning.py` orchestrates all stages, saves checkpoints to `results/{llm_id}/{model_type}/{stage_name}/checkpoints/best_model.pt`, and loads the best checkpoint from stage N before starting stage N+1.

### Patch Alignment

Time series tensors must be padded to a multiple of `PATCH_SIZE` before entering the encoder. The utility `extend_time_series_to_match_patch_size_and_aggregate()` in `time_series_datasets/util.py` handles this; it is applied inside DataLoader collation.

### Interpretability Framework

`SignalContributionTracker` (in `OpenTSLMFlamingo`) registers forward hooks that capture:
- `|x|` — residual stream magnitude before a cross-attention block
- `|x_new - x|` — contribution added by the block

`measure_signal_contribution.py` uses these hooks to produce per-layer contribution percentages.

`NoiseInjectionMixin` enables ablation studies: replacing real signals with noise and comparing model outputs measures reliance on signal vs. text context.
