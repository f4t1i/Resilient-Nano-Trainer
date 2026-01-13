# Kaggle Notebook Setup Instructions

This guide walks you through setting up and running the Resilient Nano-Trainer experiments on Kaggle with a P100 GPU.

## Prerequisites

- Kaggle account (free)
- P100 GPU access (available on free tier)
- Internet enabled in notebook settings

## Step-by-Step Setup

### Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com)
2. Click **"Create"** → **"New Notebook"**
3. Choose **"Python"** as the language
4. Name your notebook (e.g., "Resilient Nano-Trainer Experiments")

### Step 2: Enable GPU and Internet

1. Click the **three dots (⋮)** in the top-right corner
2. Select **"Settings"**
3. Under **"Accelerator"**, select **"GPU P100"**
4. Under **"Internet"**, toggle **"On"**
5. Click **"Save"**

### Step 3: Clone the Repository

In the first code cell, paste and run:

```python
# Clone the repository
!git clone https://github.com/f4t1i/Resilient-Nano-Trainer.git
%cd Resilient-Nano-Trainer

# Verify the structure
!ls -la
```

### Step 4: Install Dependencies

In the next cell:

```python
# Install dependencies
!pip install -r requirements.txt

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Running Experiments

### Experiment 1: Baseline (Standard AdamW)

```python
import subprocess
import os

os.chdir('/kaggle/working/Resilient-Nano-Trainer')

# Run baseline training
result = subprocess.run([
    'python', 'train_baseline.py',
    '--optimizer', 'baseline',
    '--seed', '42',
    '--max-iters', '5000',
    '--shift-step', '1500',
    '--run-name', 'baseline_kaggle'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

**Expected duration:** 30-45 minutes on P100

### Experiment 2: ARS with Entropy Guard Only (+Ψ)

```python
# Run ARS with Entropy Guard only
result = subprocess.run([
    'python', 'train_baseline.py',
    '--optimizer', 'ars',
    '--seed', '42',
    '--max-iters', '5000',
    '--shift-step', '1500',
    '--run-name', 'ars_psi_only_kaggle'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

**Expected duration:** 30-45 minutes on P100 (minimal overhead)

### Experiment 3: Full ARS (+Ψ +χ)

```python
# Run full ARS with Entropy Guard and Chronos-Jitter
result = subprocess.run([
    'python', 'train_baseline.py',
    '--optimizer', 'ars',
    '--seed', '42',
    '--max-iters', '5000',
    '--shift-step', '1500',
    '--run-name', 'ars_full_kaggle'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

**Expected duration:** 30-45 minutes on P100

## Analyzing Results

After all experiments complete, analyze and visualize:

```python
# Analyze results
result = subprocess.run([
    'python', 'analyze_results.py',
    '--runs', 'baseline_kaggle', 'ars_psi_only_kaggle', 'ars_full_kaggle',
    '--output-dir', 'analysis'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

### View Generated Plots

```python
from IPython.display import Image, display
from pathlib import Path

analysis_dir = Path('analysis')

# Display loss comparison
if (analysis_dir / 'loss_comparison.png').exists():
    print("Loss Comparison:")
    display(Image(str(analysis_dir / 'loss_comparison.png')))

# Display ARS metrics
if (analysis_dir / 'ars_metrics.png').exists():
    print("\nARS Metrics:")
    display(Image(str(analysis_dir / 'ars_metrics.png')))

# Display summary
if (analysis_dir / 'summary.txt').exists():
    print("\nSummary:")
    with open(analysis_dir / 'summary.txt', 'r') as f:
        print(f.read())
```

## Complete Notebook Template

Here's a complete notebook you can copy-paste:

```python
# ============================================================================
# CELL 1: Setup
# ============================================================================

!git clone https://github.com/f4t1i/Resilient-Nano-Trainer.git
%cd Resilient-Nano-Trainer
!pip install -r requirements.txt

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================================================
# CELL 2: Run Baseline
# ============================================================================

import subprocess
import os

os.chdir('/kaggle/working/Resilient-Nano-Trainer')

print("=" * 80)
print("RUNNING BASELINE EXPERIMENT")
print("=" * 80)

result = subprocess.run([
    'python', 'train_baseline.py',
    '--optimizer', 'baseline',
    '--seed', '42',
    '--max-iters', '5000',
    '--shift-step', '1500',
    '--run-name', 'baseline_kaggle'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# ============================================================================
# CELL 3: Run ARS (+Ψ)
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING ARS EXPERIMENT (+Ψ ONLY)")
print("=" * 80)

result = subprocess.run([
    'python', 'train_baseline.py',
    '--optimizer', 'ars',
    '--seed', '42',
    '--max-iters', '5000',
    '--shift-step', '1500',
    '--run-name', 'ars_psi_only_kaggle'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# ============================================================================
# CELL 4: Run Full ARS (+Ψ +χ)
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING FULL ARS EXPERIMENT (+Ψ +χ)")
print("=" * 80)

result = subprocess.run([
    'python', 'train_baseline.py',
    '--optimizer', 'ars',
    '--seed', '42',
    '--max-iters', '5000',
    '--shift-step', '1500',
    '--run-name', 'ars_full_kaggle'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# ============================================================================
# CELL 5: Analyze Results
# ============================================================================

print("\n" + "=" * 80)
print("ANALYZING RESULTS")
print("=" * 80)

result = subprocess.run([
    'python', 'analyze_results.py',
    '--runs', 'baseline_kaggle', 'ars_psi_only_kaggle', 'ars_full_kaggle',
    '--output-dir', 'analysis'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# ============================================================================
# CELL 6: Visualize Results
# ============================================================================

from IPython.display import Image, display
from pathlib import Path

analysis_dir = Path('analysis')

if (analysis_dir / 'loss_comparison.png').exists():
    print("LOSS COMPARISON")
    print("=" * 80)
    display(Image(str(analysis_dir / 'loss_comparison.png')))

if (analysis_dir / 'ars_metrics.png').exists():
    print("\nARS METRICS")
    print("=" * 80)
    display(Image(str(analysis_dir / 'ars_metrics.png')))

if (analysis_dir / 'summary.txt').exists():
    print("\nSUMMARY STATISTICS")
    print("=" * 80)
    with open(analysis_dir / 'summary.txt', 'r') as f:
        print(f.read())
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce `batch_size` in `train_baseline.py` (line ~60)
2. Reduce `max_iters` (use `--max-iters 2000`)
3. Reduce model size (`n_layer`, `n_embd`)

### Slow Training

If training is very slow:

1. Check that GPU is being used: `torch.cuda.is_available()` should return `True`
2. Verify P100 is selected in notebook settings
3. Check Kaggle's GPU availability (sometimes queued)

### Data Download Issues

If TinyShakespeare fails to download:

1. Check internet is enabled in settings
2. Try manually downloading and uploading to Kaggle datasets
3. Use a smaller dataset or synthetic data

## Monitoring Training

To monitor training in real-time, add this to a cell:

```python
import time
import json
from pathlib import Path

# Monitor the latest run
runs_dir = Path('runs')
latest_run = max(runs_dir.glob('*'), key=lambda p: p.stat().st_mtime)

print(f"Monitoring: {latest_run.name}")

while True:
    results_file = latest_run / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        train_losses = results['train_losses']
        print(f"Steps: {len(train_losses)} | Latest Loss: {train_losses[-1]:.4f}")
    
    time.sleep(10)
```

## Next Steps

1. **Analyze results** using the visualization tools
2. **Compare metrics** across the three optimizer configurations
3. **Adjust hyperparameters** based on results
4. **Document findings** in the project

## Expected Results

| Metric | Baseline | +Ψ | +Ψ +χ |
|--------|----------|-----|-------|
| Peak Loss After Shift | ~4.5 | ~3.2 | ~2.1 |
| Recovery Time (steps) | 1500+ | 800-1000 | 400-600 |
| Final Loss | ~2.8 | ~2.5 | ~2.3 |

## Support

For issues or questions:

1. Check the README.md in the repository
2. Review MEMO.md for theoretical background
3. Check the training output for error messages

---

**Happy training! 🚀**
