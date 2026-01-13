# nanoGPT + Adaptive Resonance Suppression

This fork extends [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) with **ARS** - a stability mechanism for budget-constrained training.

## New Files

- `ars_optimizer.py` - ARS optimizer wrapper
- `MEMO.md` - Design documentation
- `README_ARS.md` - This file

## Quick Start

```python
from ars_optimizer import ARSOptimizer
import torch.optim as optim

base_opt = optim.AdamW(model.parameters(), lr=1e-3)
optimizer = ARSOptimizer(base_opt)

for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step(loss.item())
    optimizer.zero_grad()
```

## Original nanoGPT

All original functionality preserved. See main README.md for nanoGPT documentation.
