# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Monte Carlo Signal Perturbation Uncertainty (MCSPU) scorer.

Computes a per-sample signal uncertainty score:
    U_signal(x) = (1/N) * sum_i  KL( p(y | x_s, x_t) || p(y | x_s + eps_i, x_t) )

where eps_i ~ N(0, sigma^2) is additive noise.  The text description
(time_series_text) is intentionally left unchanged so only signal content
is perturbed, not the statistical summary in the prompt.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


def add_additive_noise(
    sample: Dict[str, Any],
    sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Return a deep copy of *sample* with additive Gaussian noise applied to
    ``sample["time_series"]``.  ``time_series_text`` is NOT modified so the
    statistical summary in the prompt stays anchored to the original signal.

    Args:
        sample: Standard OpenTSLM sample dict.
        sigma: Standard deviation of the additive noise (eps ~ N(0, sigma^2)).
        rng: numpy random Generator.  If None a new default_rng() is created.

    Returns:
        New sample dict; ``time_series`` values have eps added, everything
        else is a deep copy of the original.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = copy.deepcopy(sample)
    ts = noisy["time_series"]

    if isinstance(ts, torch.Tensor):
        # Post-collation: shape (n_series, padded_len)
        noise = torch.tensor(
            rng.normal(0.0, sigma, ts.shape), dtype=ts.dtype, device=ts.device
        )
        noisy["time_series"] = ts + noise
    elif isinstance(ts, (list, tuple)):
        # Pre-collation: List[List[float]] or List[np.ndarray]
        new_ts = []
        for channel in ts:
            arr = np.asarray(channel, dtype=np.float64)
            noise = rng.normal(0.0, sigma, arr.shape)
            new_ts.append((arr + noise).tolist())
        noisy["time_series"] = new_ts
    else:
        raise TypeError(f"Unsupported time_series type: {type(ts)}")

    return noisy


def add_missing_data(
    sample: Dict[str, Any],
    missing_fraction: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Return a deep copy of *sample* with a random fraction of timepoints zeroed
    independently per channel.  Each call produces a different mask, so N calls
    with the same fraction give N Monte Carlo draws over missing-data patterns.

    Args:
        sample: Standard OpenTSLM sample dict.
        missing_fraction: Fraction of timepoints to zero out, in [0, 1].
        rng: numpy random Generator.  If None a new default_rng() is created.

    Returns:
        New sample dict; ``time_series`` has ``missing_fraction`` of each
        channel's timepoints set to 0.0, everything else is a deep copy.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError(f"missing_fraction must be in [0, 1], got {missing_fraction}")

    masked = copy.deepcopy(sample)
    ts = masked["time_series"]

    if isinstance(ts, torch.Tensor):
        n_channels, length = ts.shape
        result = ts.clone()
        n_mask = int(round(missing_fraction * length))
        if n_mask > 0:
            for ch in range(n_channels):
                idx = rng.choice(length, size=n_mask, replace=False)
                result[ch, idx] = 0.0
        masked["time_series"] = result
    elif isinstance(ts, (list, tuple)):
        new_ts = []
        for channel in ts:
            arr = np.asarray(channel, dtype=np.float64).copy()
            n_mask = int(round(missing_fraction * len(arr)))
            if n_mask > 0:
                idx = rng.choice(len(arr), size=n_mask, replace=False)
                arr[idx] = 0.0
            new_ts.append(arr.tolist())
        masked["time_series"] = new_ts
    else:
        raise TypeError(f"Unsupported time_series type: {type(ts)}")

    return masked


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """KL(p || q) in nats, with epsilon smoothing to avoid log(0)."""
    p = p.float()
    q = q.float()
    return float(torch.sum(p * (torch.log(p + eps) - torch.log(q + eps))).item())


class MCSpUScorer:
    """
    Monte Carlo Signal Perturbation Uncertainty scorer.

    For each test sample, runs N+1 forward passes:
      - 1 clean pass to get p_0 = p(y | x_s, x_t)
      - N perturbed passes to get p_i = p(y | perturbed(x_s), x_t)

    Supported perturbation types:
      - "gaussian"  — additive Gaussian noise, controlled by ``sigma``
      - "missing"   — random timepoint masking per channel, controlled by
                      ``missing_fraction`` (fraction of timepoints zeroed)

    Reports U_signal = mean_i KL(p_0 || p_i) as the per-sample uncertainty.

    The model must implement ``compute_class_logprobs(sample, answer_vocab)``
    (see TimeSeriesLLM base class).
    """

    def __init__(
        self,
        model,
        answer_vocab: List[str],
        n_samples: int = 50,
        sigma: float = 1.0,
        missing_fraction: float = 0.5,
        perturbation_type: str = "gaussian",
        seed: Optional[int] = None,
        class_batch_size: int = 4,
    ):
        if perturbation_type not in ("gaussian", "missing"):
            raise ValueError(f"perturbation_type must be 'gaussian' or 'missing', got {perturbation_type!r}")
        self.model = model
        self.answer_vocab = answer_vocab
        self.n_samples = n_samples
        self.sigma = sigma
        self.missing_fraction = missing_fraction
        self.perturbation_type = perturbation_type
        self.class_batch_size = class_batch_size
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_sample(
        self,
        sample: Dict[str, Any],
        answer_vocab_override: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute MCSPU score for a single sample.

        Args:
            sample: Standard OpenTSLM sample dict (pre-collation form).
            answer_vocab_override: If given, use this vocab instead of
                ``self.answer_vocab`` (needed for ECG QA's per-sample vocab).

        Returns:
            Dict with keys:
                mcspu_score      float   mean KL divergence U_signal
                clean_logprobs   list    raw log-probs over answer_vocab (clean signal)
                clean_probs      list    softmax probabilities (clean signal)
                clean_pred       str     argmax class on clean signal
                per_noise_kl     list    KL divergence for each of the N realizations
                sigma            float   noise std used
                n_samples        int     N used
                answer_vocab     list    the answer vocab used
        """
        vocab = answer_vocab_override if answer_vocab_override is not None else self.answer_vocab

        # --- clean pass ---
        clean_logprobs = self._score(sample, vocab)
        p0 = torch.softmax(clean_logprobs.float(), dim=0)

        # --- N perturbed passes ---
        per_kl: List[float] = []
        for _ in range(self.n_samples):
            if self.perturbation_type == "gaussian":
                perturbed = add_additive_noise(sample, sigma=self.sigma, rng=self._rng)
            else:
                perturbed = add_missing_data(sample, missing_fraction=self.missing_fraction, rng=self._rng)
            perturbed_logprobs = self._score(perturbed, vocab)
            p_i = torch.softmax(perturbed_logprobs.float(), dim=0)
            per_kl.append(_kl_divergence(p0, p_i))

        mcspu = float(np.mean(per_kl))
        clean_pred = vocab[int(torch.argmax(p0).item())]

        return {
            "mcspu_score": mcspu,
            "clean_logprobs": clean_logprobs.tolist(),
            "clean_probs": p0.tolist(),
            "clean_pred": clean_pred,
            "per_noise_kl": per_kl,
            "perturbation_type": self.perturbation_type,
            "sigma": self.sigma if self.perturbation_type == "gaussian" else None,
            "missing_fraction": self.missing_fraction if self.perturbation_type == "missing" else None,
            "n_samples": self.n_samples,
            "answer_vocab": vocab,
        }

    def score_dataset(
        self,
        dataset,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score all (or up to ``max_samples``) items from *dataset*.

        Adds ``sample_idx`` and ``ground_truth`` fields to each result dict.
        Prints a progress line every 10 samples.
        """
        results = []
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        for idx in range(n):
            sample = dataset[idx]
            ground_truth = sample.get("answer", "")

            # Per-sample vocab for ECG QA (stored by ECGQACoTQADataset)
            vocab_override = sample.get("possible_answers", None)

            result = self.score_sample(sample, answer_vocab_override=vocab_override)
            result["sample_idx"] = idx
            result["ground_truth"] = ground_truth
            results.append(result)

            if (idx + 1) % 10 == 0 or (idx + 1) == n:
                print(f"  [{idx + 1}/{n}]  mcspu={result['mcspu_score']:.4f}  pred={result['clean_pred']!r}  gt={ground_truth!r}")

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(self, sample: Dict[str, Any], vocab: List[str]) -> torch.Tensor:
        """Collate sample and call model.compute_class_logprobs."""
        collated_batch = extend_time_series_to_match_patch_size_and_aggregate(
            [copy.deepcopy(sample)]
        )
        collated = collated_batch[0]
        self.model.eval()
        with torch.no_grad():
            return self.model.compute_class_logprobs(
                collated, vocab, class_batch_size=self.class_batch_size
            )
