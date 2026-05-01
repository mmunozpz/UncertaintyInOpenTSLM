# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np


class NoiseInjectionMixin:
    """
    Mixin providing noise injection for interpretability testing.

    Add to any QADataset subclass to enable replacing real time series signals
    with noise during training/evaluation. This tests whether the model actually
    uses the signal content or relies on other features (text prompts, metadata).

    Each concrete subclass gets its OWN class-level state due to Python's
    class attribute resolution: setting cls._use_noise on a subclass
    creates that attribute on the subclass, not on the mixin.

    Usage:
        class MyDataset(NoiseInjectionMixin, QADataset):
            @classmethod
            def clear_caches(cls):
                # Must clear QADataset's loaded flag + cached splits
                ...

            def _get_text_time_series_prompt_list(self, row):
                # ... normalize signal ...
                if self.__class__._use_noise:
                    signal = self.__class__._generate_noise_signal(len(signal), ...)
                # ... return TextTimeSeriesPrompt with original mean/std text ...
    """

    _use_noise = False
    _noise_type = "gaussian"  # Options: "gaussian", "shuffle", "zero", "uniform"
    _noise_seed = None  # For reproducibility
    _strip_stats = False  # If True, remove mean/std from text descriptions

    @classmethod
    def set_noise_mode(cls, use_noise: bool, noise_type: str = "gaussian", noise_seed: int = None, strip_stats: bool = False):
        """
        Set noise mode for all dataset instances (for interpretability testing).

        This replaces real signals with noise to test if the model actually
        uses the signal content or relies on other features (text prompts, metadata).

        Args:
            use_noise: If True, replace real signals with noise
            noise_type: Type of noise ("gaussian", "shuffle", "zero", "uniform")
            noise_seed: Random seed for reproducibility (set ONCE here, not per-signal)
            strip_stats: If True, remove mean/std summary statistics from text descriptions
        """
        # Clear all caches to ensure noise mode is applied fresh
        # Each dataset subclass must define clear_caches()
        cls.clear_caches()

        cls._use_noise = use_noise
        cls._noise_type = noise_type
        cls._noise_seed = noise_seed
        cls._strip_stats = strip_stats
        # Set seed ONCE here, so subsequent calls generate different sequences
        if noise_seed is not None:
            np.random.seed(noise_seed)
        if use_noise:
            stats_msg = " (stats stripped from text)" if strip_stats else " (stats kept in text)"
            print(f"[NOISE MODE] {cls.__name__}: signals will be replaced with '{noise_type}' noise (seed={noise_seed}){stats_msg}")

    @classmethod
    def get_noise_mode(cls) -> dict:
        """Get current noise configuration."""
        return {
            "use_noise": cls._use_noise,
            "noise_type": cls._noise_type,
            "noise_seed": cls._noise_seed,
            "strip_stats": cls._strip_stats,
        }

    @classmethod
    def _generate_noise_signal(cls, length: int, noise_type: str, original_signal: np.ndarray = None) -> np.ndarray:
        """
        Generate a noise signal of the specified type.

        Args:
            length: Length of the signal to generate
            noise_type: Type of noise ("gaussian", "shuffle", "zero", "uniform")
            original_signal: Original signal (required for "shuffle" noise type)

        Returns:
            Noise signal as numpy array

        Note: The random seed is set ONCE in set_noise_mode(), not here.
              This ensures each signal gets unique noise while still being reproducible.
        """
        if noise_type == "gaussian":
            # Standard Gaussian noise (mean=0, std=1)
            return np.random.randn(length)
        elif noise_type == "shuffle":
            # Shuffle the original signal (destroys temporal structure but preserves amplitude distribution)
            if original_signal is None:
                return np.random.randn(length)
            shuffled = original_signal.copy()
            np.random.shuffle(shuffled)
            return shuffled
        elif noise_type == "zero":
            # All zeros (no signal)
            return np.zeros(length)
        elif noise_type == "uniform":
            # Uniform random noise in [-1, 1]
            return np.random.uniform(-1, 1, length)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}. Options: gaussian, shuffle, zero, uniform")
