# Adaptive Resonance Suppression (ARS) - Design Memo

## Problem: Narcissus-Failure

Training instability after distribution shifts causes oscillation, divergence, and wasted compute.

## Solution: Three-Layer Defense

1. **Entropy Guard (Ψ_t)**: Detects periodicity via autocorrelation
2. **Surprise Gate (Φ_t)**: Adaptive damping based on loss surprise  
3. **Chronos-Jitter (χ_t)**: Noise injection to break phase-lock

## Expected Impact

- Peak weight norm: 6.82 → 1.21 (~82% reduction)
- Recovery time: 1500+ → 400-600 steps
- Cost savings: 20-40% via reduced wasted compute

See `ars_optimizer.py` for implementation.
