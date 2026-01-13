# nanoGPT + Adaptive Resonance Suppression (ARS)

This is a fork of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) extended with **Adaptive Resonance Suppression (ARS)**, a novel optimization technique that prevents training instability and reduces "wasted compute" during neural network training.

## What's New: Adaptive Resonance Suppression

ARS addresses the **Narcissus-Failure** problem: a catastrophic training instability where models enter self-oscillating states after data distribution shifts, leading to divergence and wasted compute.

### Key Components

1. **Entropy Guard (Ψ_t):** Detects periodicity via lag-1 autocorrelation
2. **Surprise Gate (Φ_t):** Applies adaptive momentum damping
3. **Chronos-Jitter (χ_t):** Breaks phase-lock with adaptive noise

### New Files

- **`ars_optimizer.py`** - ARS optimizer wrapper (works with any PyTorch optimizer)
- **`train_baseline.py`** - Training script with data-shift provocation for testing ARS
- **`analyze_results.py`** - Analysis and visualization tools
- **`MEMO.md`** - Complete theoretical documentation
- **`KAGGLE_INSTRUCTIONS.md`** - Step-by-step guide for running on Kaggle

## Quick Start with ARS

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib
```

### 2. Run Baseline Experiment

```bash
python train_baseline.py --optimizer baseline --seed 42 --max-iters 5000
```

### 3. Run ARS Experiment

```bash
python train_baseline.py --optimizer ars --seed 42 --max-iters 5000
```

### 4. Analyze Results

```bash
python analyze_results.py --runs baseline_* ars_* --output-dir analysis
```

## Using Original nanoGPT

All original nanoGPT functionality remains unchanged. To use the original training script:

```bash
python train.py config/train_shakespeare_char.py
```

See the [original nanoGPT README](https://github.com/karpathy/nanoGPT) for complete documentation.

## Documentation

- **[MEMO.md](MEMO.md)** - Complete ARS design and methodology
- **[KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md)** - Kaggle setup guide
- **[Original nanoGPT README](https://github.com/karpathy/nanoGPT)** - nanoGPT documentation

## Expected Results

| Metric | Baseline | ARS |
|--------|----------|-----|
| Peak Weight Norm | ~6.82 | ~1.21 |
| Divergence Events | High | Minimal |
| Compute Efficiency | Low | High |

## Citation

### ARS

```bibtex
@software{resilient_nano_trainer_2026,
  title={Adaptive Resonance Suppression for Stable Training},
  author={f4t1i and Manus Agent},
  year={2026},
  url={https://github.com/f4t1i/Resilient-Nano-Trainer}
}
```

### nanoGPT

```bibtex
@software{karpathy2022nanogpt,
  title={nanoGPT},
  author={Karpathy, Andrej},
  year={2022},
  url={https://github.com/karpathy/nanoGPT}
}
```

## License

This project inherits the MIT License from nanoGPT. See [LICENSE](LICENSE) for details.
