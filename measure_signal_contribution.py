#!/usr/bin/env python3
"""
Measure actual signal contribution in OpenTSLM-Flamingo using forward hooks.

This script runs inference on the test set and measures the real activation
magnitudes to calculate how much the ECG signal contributes to the output.
"""

import sys
import os
import argparse
from collections import defaultdict

# Set up paths before any imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))
sys.path.insert(0, os.path.join(script_dir, 'src', 'open_flamingo'))

import torch
import numpy as np
from tqdm import tqdm


class SignalContributionHook:
    """
    Forward hook to measure gated cross-attention contributions.

    Captures:
    - |x| : magnitude of residual stream (input to cross-attn block)
    - |gated_output| : magnitude of attn_output * tanh(gate)
    - contribution % : |gated_output| / |x| * 100
    """

    def __init__(self):
        self.measurements = defaultdict(list)
        self.hooks = []

    def _make_hook(self, layer_idx):
        """Create a hook for a specific layer."""
        def hook(module, inputs, output):
            # inputs[0] is x (the residual stream input)
            # output is x_new = x + gated_attn_output + gated_ff_output

            x = inputs[0]  # residual stream input
            x_new = output  # output after cross-attn block

            # The difference is the total added contribution
            # x_new = x + attn_contribution + ff_contribution
            total_contribution = x_new - x

            # Measure magnitudes (using L2 norm per token, then average)
            with torch.no_grad():
                # |x| - residual stream magnitude
                x_magnitude = x.norm(dim=-1).mean().item()

                # |contribution| - what was added
                contrib_magnitude = total_contribution.norm(dim=-1).mean().item()

                # Contribution percentage
                if x_magnitude > 1e-8:
                    contrib_pct = (contrib_magnitude / x_magnitude) * 100
                else:
                    contrib_pct = 0.0

                self.measurements[f'layer_{layer_idx}'].append({
                    'x_magnitude': x_magnitude,
                    'contrib_magnitude': contrib_magnitude,
                    'contrib_pct': contrib_pct
                })

        return hook

    def register_hooks(self, model):
        """Register hooks on all gated cross-attention blocks."""
        # Find the gated cross-attention layers
        # They're at: model.lang_encoder.gated_cross_attn_layers[i]

        if hasattr(model, 'llm'):
            lang_encoder = model.llm.lang_encoder
        elif hasattr(model, 'lang_encoder'):
            lang_encoder = model.lang_encoder
        else:
            raise ValueError("Cannot find lang_encoder in model")

        if hasattr(lang_encoder, 'gated_cross_attn_layers'):
            for idx, layer in enumerate(lang_encoder.gated_cross_attn_layers):
                hook = layer.register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)
            print(f"Registered hooks on {len(self.hooks)} gated cross-attention layers")
        else:
            # Try to find them in the decoder blocks
            if hasattr(lang_encoder, 'model') and hasattr(lang_encoder.model, 'layers'):
                for idx, layer in enumerate(lang_encoder.model.layers):
                    if hasattr(layer, 'gated_cross_attn_layer') and layer.gated_cross_attn_layer is not None:
                        hook = layer.gated_cross_attn_layer.register_forward_hook(self._make_hook(idx))
                        self.hooks.append(hook)
                print(f"Registered hooks on {len(self.hooks)} gated cross-attention layers")
            else:
                raise ValueError("Cannot find gated_cross_attn_layers in model")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_summary(self):
        """Get summary statistics across all samples."""
        summary = {}

        for layer_name, measurements in self.measurements.items():
            x_mags = [m['x_magnitude'] for m in measurements]
            contrib_mags = [m['contrib_magnitude'] for m in measurements]
            contrib_pcts = [m['contrib_pct'] for m in measurements]

            summary[layer_name] = {
                'x_magnitude_mean': np.mean(x_mags),
                'x_magnitude_std': np.std(x_mags),
                'contrib_magnitude_mean': np.mean(contrib_mags),
                'contrib_magnitude_std': np.std(contrib_mags),
                'contrib_pct_mean': np.mean(contrib_pcts),
                'contrib_pct_std': np.std(contrib_pcts),
                'n_samples': len(measurements)
            }

        # Overall average across layers
        all_x_mags = []
        all_contrib_pcts = []
        for layer_name, measurements in self.measurements.items():
            all_x_mags.extend([m['x_magnitude'] for m in measurements])
            all_contrib_pcts.extend([m['contrib_pct'] for m in measurements])

        summary['overall'] = {
            'x_magnitude_mean': np.mean(all_x_mags),
            'contrib_pct_mean': np.mean(all_contrib_pcts),
            'contrib_pct_std': np.std(all_contrib_pcts),
            'n_measurements': len(all_contrib_pcts)
        }

        return summary

    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()


def load_model_and_tokenizer(checkpoint_path, llm_id="meta-llama/Llama-3.2-1B", device="cuda"):
    """Load OpenTSLMFlamingo model from checkpoint."""
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

    print(f"Loading model from {checkpoint_path}")

    # Initialize model (matching curriculum_learning.py)
    model = OpenTSLMFlamingo(
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
        llm_id=llm_id,
        device=device,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'llm' in checkpoint:
        state_dict = checkpoint['llm']
    else:
        state_dict = checkpoint

    # Filter and load state dict
    model_state = model.state_dict()
    filtered_state = {}
    for key, value in state_dict.items():
        # Remove prefixes if present
        clean_key = key
        for prefix in ['llm.', 'model.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        if clean_key in model_state:
            filtered_state[clean_key] = value
        elif key in model_state:
            filtered_state[key] = value

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    return model


def run_measurement(checkpoint_path, max_samples=100, device='cuda'):
    """Run measurement on a checkpoint."""
    from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
    from torch.utils.data import DataLoader

    # Load model
    model = load_model_and_tokenizer(checkpoint_path, device=device)

    # Create hook
    hook = SignalContributionHook()
    hook.register_hooks(model)

    # Load dataset
    ECGQACoTQADataset.set_noise_mode(use_noise=False)
    dataset = ECGQACoTQADataset(
        split="test",
        EOS_TOKEN=model.tokenizer.eos_token,
        max_samples=max_samples,
        preload_processed_data=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=model.collate_fn
    )

    print(f"Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Measuring"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass (hooks will capture measurements)
            try:
                _ = model.compute_loss(batch)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue

    # Get summary
    summary = hook.get_summary()

    # Cleanup
    hook.remove_hooks()
    del model
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Measure signal contribution in OpenTSLM-Flamingo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    summary = run_measurement(args.checkpoint, args.max_samples, args.device)

    print("\n" + "=" * 60)
    print("SIGNAL CONTRIBUTION SUMMARY")
    print("=" * 60)

    print(f"\nOverall (across all layers and samples):")
    print(f"  Residual stream magnitude: {summary['overall']['x_magnitude_mean']:.4f}")
    print(f"  Signal contribution: {summary['overall']['contrib_pct_mean']:.4f}% ± {summary['overall']['contrib_pct_std']:.4f}%")
    print(f"  N measurements: {summary['overall']['n_measurements']}")

    print(f"\nPer-layer breakdown:")
    for layer_name in sorted(summary.keys()):
        if layer_name == 'overall':
            continue
        s = summary[layer_name]
        print(f"  {layer_name}: {s['contrib_pct_mean']:.4f}% ± {s['contrib_pct_std']:.4f}%")


if __name__ == "__main__":
    main()
