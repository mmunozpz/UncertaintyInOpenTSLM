# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Unit tests for MCSPU (no GPU or checkpoint required)."""

import sys
import copy

import numpy as np
import torch

sys.path.insert(0, "src")

from opentslm.uncertainty.mcspu import add_additive_noise, MCSpUScorer

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(n_channels: int = 1, length: int = 8) -> dict:
    """Minimal sample dict in pre-collation form."""
    return {
        "pre_prompt": "Classify.",
        "post_prompt": "Answer:",
        "time_series_text": ["mean 0.00 std 1.00"],
        "time_series": [[float(i) for i in range(length)] for _ in range(n_channels)],
        "answer": "A",
    }


class _RandomModel:
    """Mock model: returns random log-probs regardless of input."""
    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def eval(self):
        return self

    def compute_class_logprobs(self, sample, answer_vocab):
        return torch.tensor(
            self._rng.standard_normal(len(answer_vocab)), dtype=torch.float32
        )


class _ConstantModel:
    """Mock model: always concentrates probability on the first class."""
    def eval(self):
        return self

    def compute_class_logprobs(self, sample, answer_vocab):
        # Logits heavily favour class 0 → deterministic after softmax.
        logits = torch.zeros(len(answer_vocab))
        logits[0] = 100.0
        return logits


# ---------------------------------------------------------------------------
# Test add_additive_noise
# ---------------------------------------------------------------------------

print("\n=== add_additive_noise ===")

sample = _make_sample(n_channels=2, length=8)
rng = np.random.default_rng(42)
noisy = add_additive_noise(sample, sigma=0.5, rng=rng)

original_ts = [list(ch) for ch in sample["time_series"]]

check(
    "time_series_text unchanged",
    noisy["time_series_text"] == sample["time_series_text"],
)
check(
    "original sample not mutated",
    sample["time_series"] == original_ts,
)
check(
    "answer unchanged",
    noisy["answer"] == sample["answer"],
)
noise_applied_ch0 = np.array(noisy["time_series"][0]) - np.array(original_ts[0])
check(
    "noise is non-zero",
    not np.allclose(noise_applied_ch0, 0),
)
check(
    "shape preserved",
    len(noisy["time_series"]) == 2 and len(noisy["time_series"][0]) == 8,
)

# Different rng calls give different noise
noisy2 = add_additive_noise(sample, sigma=0.5, rng=rng)
check(
    "independent calls give different noise",
    not np.allclose(
        np.array(noisy["time_series"][0]),
        np.array(noisy2["time_series"][0]),
    ),
)

# Tensor form (post-collation)
sample_tensor = copy.deepcopy(sample)
sample_tensor["time_series"] = torch.tensor(sample["time_series"], dtype=torch.float32)
noisy_t = add_additive_noise(sample_tensor, sigma=1.0)
check(
    "tensor form: output is a tensor",
    isinstance(noisy_t["time_series"], torch.Tensor),
)
check(
    "tensor form: noise applied",
    not torch.allclose(noisy_t["time_series"], sample_tensor["time_series"]),
)

# ---------------------------------------------------------------------------
# Test MCSpUScorer smoke
# ---------------------------------------------------------------------------

print("\n=== MCSpUScorer smoke ===")

vocab = ["A", "B", "C", "D"]
model = _RandomModel(seed=7)
scorer = MCSpUScorer(model, vocab, n_samples=5, sigma=1.0, seed=0)
sample = _make_sample()

result = scorer.score_sample(sample)

check("mcspu_score key present", "mcspu_score" in result)
check("clean_probs key present", "clean_probs" in result)
check("clean_pred key present", "clean_pred" in result)
check("per_noise_kl key present", "per_noise_kl" in result)
check("mcspu_score >= 0", result["mcspu_score"] >= 0, f"got {result['mcspu_score']}")
check("per_noise_kl length == n_samples", len(result["per_noise_kl"]) == 5)
check("clean_pred in vocab", result["clean_pred"] in vocab)
check(
    "clean_probs sum ~1",
    abs(sum(result["clean_probs"]) - 1.0) < 1e-4,
    f"sum={sum(result['clean_probs'])}",
)
check("sigma stored", result["sigma"] == 1.0)
check("n_samples stored", result["n_samples"] == 5)
check("answer_vocab stored", result["answer_vocab"] == vocab)

# ---------------------------------------------------------------------------
# Test KL = 0 for a deterministic (constant) model
# ---------------------------------------------------------------------------

print("\n=== Deterministic model: KL ~= 0 ===")

det_scorer = MCSpUScorer(_ConstantModel(), ["X", "Y", "Z"], n_samples=10, sigma=5.0, seed=1)
det_result = det_scorer.score_sample(_make_sample())

check(
    "KL ~ 0 for deterministic model",
    abs(det_result["mcspu_score"]) < 1e-4,
    f"got {det_result['mcspu_score']}",
)

# ---------------------------------------------------------------------------
# Test answer_vocab_override
# ---------------------------------------------------------------------------

print("\n=== answer_vocab_override ===")

scorer2 = MCSpUScorer(_RandomModel(), ["A", "B"], n_samples=3, sigma=0.1, seed=2)
override_result = scorer2.score_sample(_make_sample(), answer_vocab_override=["X", "Y", "Z"])

check(
    "override vocab used",
    override_result["answer_vocab"] == ["X", "Y", "Z"],
)
check(
    "clean_probs length matches override",
    len(override_result["clean_probs"]) == 3,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print(f"{'='*50}")

import sys as _sys
_sys.exit(1 if FAIL > 0 else 0)
