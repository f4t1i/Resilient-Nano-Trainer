# Experimental Results

This directory contains all experimental results, visualizations, and data from the Adaptive Resonance Suppression (ARS) experiments.

## Files

- **`experimental_results.json`**: Complete experimental data in structured JSON format
- **`comprehensive_comparison.png`**: Main comparison figure showing all experiments
- **`ars_mechanism.png`**: Visualization of ARS internal dynamics

## Summary

### Experimental Setup

- **Platform**: Kaggle P100 GPU
- **Dataset**: Shakespeare Character-Level (1MB text corpus)
- **Model**: GPT (4 layers, 4 heads, 256 embedding, 3.23M parameters)
- **Data Shift**: Text reversal at step 300 (extreme distribution shift)
- **Training Steps**: 1000 total (700 post-shift)

### Results

| Experiment | Optimizer | Divergence | Survival | Final Loss | Status |
|------------|-----------|------------|----------|------------|--------|
| Baseline | AdamW | Step 650 | 350 steps | 2.099 | Diverged |
| ARS (Original) | ARS + AdamW (α=2.0) | Step 400 | 100 steps | 2.745 | Diverged |
| ARS (Optimized) | ARS + AdamW (α=1.0) | None | 700+ steps | 1.935 | **Stable** |

### Key Findings

1. **Stability Improvement**: ARS (optimized) survived 2x longer than baseline without diverging
2. **Parameter Tuning**: Reducing alpha from 2.0 to 1.0 and increasing phi_min from 0.1 to 0.3 prevented premature divergence
3. **Loss Improvement**: ARS achieved 7.8% better final loss despite the extreme data shift
4. **Weight Norm Control**: ARS kept maximum weight norm at 99.14, just below the divergence threshold of 100

### ARS Mechanism Performance

The Entropy Guard successfully detected resonance patterns:
- **Minimum Ψ_t**: 0.112 (detected at step 50)
- **Maximum |ρ₁|**: 0.888 (high autocorrelation indicating periodicity)
- **Average Φ_t**: 0.978 (gentle intervention, 97.8% gradient flow maintained)

This demonstrates that ARS can detect and mitigate instability without over-damping the training process.

## Replication

To replicate these results:

```bash
# Run baseline
python baseline_experiment.py

# Run optimized ARS
python ars_tuned_experiment.py

# Generate plots
python generate_plots.py
```

All experiments use the same random seed (1337) for reproducibility.
