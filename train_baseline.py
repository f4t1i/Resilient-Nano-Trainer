"""
Baseline Training Script with Data-Shift Provocation

This script trains a small nanoGPT model on TinyShakespeare and deliberately
provokes a data distribution shift to trigger the Narcissus-Failure.

The shift is implemented by reversing the text after a certain number of steps.
This creates a dramatic change in the data distribution that can cause training
instability if not handled properly.

Usage:
    python train_baseline.py --optimizer baseline --seed 42
    python train_baseline.py --optimizer ars --seed 42
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ars_optimizer import ARSOptimizer, ARSOptimizerFactory


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration."""
    
    # Model
    n_layer = 6
    n_head = 6
    n_embd = 384
    block_size = 256
    bias = False
    dropout = 0.1
    
    # Training
    batch_size = 64
    learning_rate = 6e-4
    max_iters = 10000
    eval_interval = 500
    eval_iters = 200
    
    # Data shift
    shift_step = 2000  # When to apply the data shift
    shift_enabled = True
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_interval = 100
    save_interval = 1000
    
    # ARS Configuration
    ars_config = {
        'window_size': 100,
        'alpha': 1.0,
        'phi_min': 0.1,
        'epsilon': 1e-6,
        'jitter_scale': 0.1,
        'enable_entropy_guard': True,
        'enable_jitter': True,
        'surprise_metric': 'loss',
    }


# ============================================================================
# Simple GPT Model (Minimal Implementation)
# ============================================================================

class GPTConfig:
    """GPT model configuration."""
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class Head(torch.nn.Module):
    """Single attention head."""
    
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = head_size ** -0.5
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        out = scores @ v
        return out


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention."""
    
    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = torch.nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(torch.nn.Module):
    """Feed-forward network."""
    
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    """Transformer block."""
    
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, n_embd, block_size, dropout)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(torch.nn.Module):
    """Minimal GPT model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = torch.nn.Embedding(config.block_size, config.n_embd)
        self.blocks = torch.nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.ln_f = torch.nn.LayerNorm(config.n_embd)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size):
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_path: str = "data/tiny_shakespeare.txt") -> str:
    """Load or download TinyShakespeare dataset."""
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Downloading TinyShakespeare to {data_path}...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def create_dataset(text: str, block_size: int, batch_size: int, split: float = 0.9):
    """Create train/val datasets."""
    
    # Character-level tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Encode text
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split
    n = int(len(data) * split)
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(data, batch_size, block_size, device):
        """Get a random batch."""
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix])
        return x.to(device), y.to(device)
    
    return train_data, val_data, vocab_size, encode, decode, get_batch, stoi, itos


# ============================================================================
# Training Loop
# ============================================================================

def train(
    config: Config,
    optimizer_type: str = "baseline",
    seed: int = 42,
    run_name: Optional[str] = None,
):
    """Main training loop."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{optimizer_type}_{timestamp}"
    
    # Create output directory
    output_dir = Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    text = load_data()
    train_data, val_data, vocab_size, encode, decode, get_batch, stoi, itos = \
        create_dataset(text, config.block_size, config.batch_size)
    
    # Create model
    print(f"Creating model on {config.device}...")
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    model = GPT(gpt_config).to(config.device)
    
    # Create optimizer
    print(f"Creating {optimizer_type} optimizer...")
    if optimizer_type == "baseline":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
    elif optimizer_type == "ars":
        optimizer = ARSOptimizerFactory.create_ars_adamw(
            model,
            lr=config.learning_rate,
            ars_config=config.ars_config,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Training state
    train_losses = []
    val_losses = []
    metrics_log = []
    
    print(f"Starting training for {config.max_iters} iterations...")
    print(f"Data shift enabled: {config.shift_enabled} (at step {config.shift_step})")
    print("-" * 80)
    
    # Training loop
    for step in range(config.max_iters):
        
        # Get batch
        x, y = get_batch(train_data, config.batch_size, config.block_size, config.device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        if optimizer_type == "ars":
            optimizer.step(loss.item(), model)
        else:
            optimizer.step()
        
        # Logging
        train_losses.append(loss.item())
        
        # Get metrics if using ARS
        if optimizer_type == "ars":
            metrics = optimizer.get_metrics()
            metrics_log.append({
                'step': step,
                'loss': loss.item(),
                **metrics,
            })
        
        # Print progress
        if (step + 1) % config.log_interval == 0:
            avg_loss = np.mean(train_losses[-config.log_interval:])
            
            if optimizer_type == "ars":
                print(f"Step {step+1:5d} | Loss: {avg_loss:.4f} | "
                      f"Φ_t: {metrics['phi_t']:.3f} | Ψ_t: {metrics['psi_t']:.3f} | "
                      f"ρ₁: {metrics['rho_1']:.3f}")
            else:
                print(f"Step {step+1:5d} | Loss: {avg_loss:.4f}")
        
        # Validation
        if (step + 1) % config.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(config.eval_iters):
                    x_val, y_val = get_batch(val_data, config.batch_size, config.block_size, config.device)
                    _, loss_val = model(x_val, y_val)
                    val_loss += loss_val.item()
            val_loss /= config.eval_iters
            val_losses.append(val_loss)
            model.train()
            
            print(f"  → Validation Loss: {val_loss:.4f}")
        
        # Data shift
        if config.shift_enabled and step == config.shift_step:
            print("\n" + "=" * 80)
            print("APPLYING DATA SHIFT (reversing text)...")
            print("=" * 80 + "\n")
            
            # Reverse the training data
            train_data = torch.flip(train_data, [0])
    
    # Save results
    print("\n" + "=" * 80)
    print("Training completed. Saving results...")
    
    # Save losses
    results = {
        'config': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_iters': config.max_iters,
            'optimizer_type': optimizer_type,
            'seed': seed,
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
    }
    
    # Save JSON results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics log if ARS
    if optimizer_type == "ars" and metrics_log:
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_log, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")
    
    print(f"Results saved to {output_dir}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    
    return results, output_dir


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nanoGPT with optional ARS optimizer")
    parser.add_argument("--optimizer", type=str, choices=["baseline", "ars"], default="baseline",
                        help="Optimizer to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for output directory")
    parser.add_argument("--max-iters", type=int, default=10000, help="Maximum iterations")
    parser.add_argument("--shift-step", type=int, default=2000, help="Step at which to apply data shift")
    parser.add_argument("--no-shift", action="store_true", help="Disable data shift")
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.max_iters = args.max_iters
    config.shift_step = args.shift_step
    config.shift_enabled = not args.no_shift
    
    # Train
    results, output_dir = train(
        config,
        optimizer_type=args.optimizer,
        seed=args.seed,
        run_name=args.run_name,
    )
