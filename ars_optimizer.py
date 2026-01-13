"""
Adaptive Resonance Suppression (ARS) Optimizer

A wrapper optimizer that detects and suppresses training instability caused by
self-oscillation (Narcissus-Failure) through adaptive damping and phase-lock breaking.

Author: Manus Agent
Date: 2026-01-13
"""

import torch
import numpy as np
from collections import deque
from typing import Optional, Dict, Any


class ARSOptimizer:
    """
    Adaptive Resonance Suppression Optimizer Wrapper.
    
    Wraps any PyTorch optimizer and applies adaptive damping based on:
    1. Entropy Guard (Ψ_t): Detects periodicity via autocorrelation
    2. Surprise Gate (Φ_t): Applies adaptive momentum damping
    3. Chronos-Jitter (χ_t): Breaks phase-lock with adaptive noise
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        window_size: int = 100,
        alpha: float = 1.0,
        phi_min: float = 0.1,
        epsilon: float = 1e-6,
        jitter_scale: float = 0.1,
        enable_entropy_guard: bool = True,
        enable_jitter: bool = True,
        surprise_metric: str = "loss",
    ):
        """
        Initialize the ARS Optimizer.
        
        Args:
            base_optimizer: The underlying PyTorch optimizer (e.g., AdamW)
            window_size: Size of the surprise history window for autocorrelation
            alpha: Sensitivity of damping to surprise (higher = more aggressive damping)
            phi_min: Minimum damping factor (0.0 to 1.0)
            epsilon: Small constant for numerical stability
            jitter_scale: Maximum magnitude of learning rate jitter
            enable_entropy_guard: Whether to enable Entropy Guard (Ψ_t)
            enable_jitter: Whether to enable Chronos-Jitter (χ_t)
            surprise_metric: Metric to use for surprise ("loss" or "gradient_norm")
        """
        self.base_optimizer = base_optimizer
        self.window_size = window_size
        self.alpha = alpha
        self.phi_min = phi_min
        self.epsilon = epsilon
        self.jitter_scale = jitter_scale
        self.enable_entropy_guard = enable_entropy_guard
        self.enable_jitter = enable_jitter
        self.surprise_metric = surprise_metric
        
        # History tracking
        self.surprise_history = deque(maxlen=window_size)
        self.step_count = 0
        
        # Store original beta values for AdamW (if applicable)
        self.original_betas = {}
        for group in self.base_optimizer.param_groups:
            if 'betas' in group:
                self.original_betas[id(group)] = group['betas']
        
        # Metrics for logging
        self.metrics = {
            'rho_1': 0.0,
            'psi_t': 1.0,
            'phi_t': 1.0,
            'jitter_t': 0.0,
            'surprise_t': 0.0,
            'surprise_tilde': 0.0,
        }
    
    def compute_surprise(self, loss: float, model: Optional[torch.nn.Module] = None) -> float:
        """
        Compute the surprise metric.
        
        Args:
            loss: The current loss value
            model: The model (used for gradient_norm metric)
        
        Returns:
            The surprise value
        """
        if self.surprise_metric == "loss":
            return loss
        elif self.surprise_metric == "gradient_norm":
            if model is None:
                raise ValueError("model required for gradient_norm surprise metric")
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += (param.grad ** 2).sum().item()
            return np.sqrt(grad_norm)
        else:
            raise ValueError(f"Unknown surprise metric: {self.surprise_metric}")
    
    def compute_entropy_guard(self) -> tuple:
        """
        Compute the Entropy Guard (Ψ_t) using lag-1 autocorrelation.
        
        Returns:
            (rho_1, psi_t): Lag-1 autocorrelation and entropy guard value
        """
        if len(self.surprise_history) < 2:
            return 0.0, 1.0
        
        history = np.array(list(self.surprise_history))
        
        # Compute lag-1 autocorrelation
        if len(history) < 2:
            rho_1 = 0.0
        else:
            # Normalize the series
            mean = np.mean(history)
            std = np.std(history)
            
            if std < self.epsilon:
                rho_1 = 0.0
            else:
                normalized = (history - mean) / std
                rho_1 = np.corrcoef(normalized[:-1], normalized[1:])[0, 1]
                
                # Handle NaN (can occur with constant signals)
                if np.isnan(rho_1):
                    rho_1 = 0.0
        
        # Compute Entropy Guard: Ψ_t = max(ε, 1 - ρ₁²)
        psi_t = max(self.epsilon, 1.0 - rho_1 ** 2)
        
        return rho_1, psi_t
    
    def compute_damping_gate(self, surprise_t: float, psi_t: float) -> float:
        """
        Compute the Surprise Gate (Φ_t) with adaptive damping.
        
        Args:
            surprise_t: Current surprise value
            psi_t: Entropy Guard value
        
        Returns:
            phi_t: Damping gate value (0.0 to 1.0)
        """
        # Effective surprise: S̃_t = S_t / Ψ_t
        surprise_tilde = surprise_t / psi_t if psi_t > self.epsilon else surprise_t
        
        # Damping gate (Variant B): Φ_t = Φ_min + (1 - Φ_min) · exp(-α · S̃_t)
        phi_t = self.phi_min + (1.0 - self.phi_min) * np.exp(-self.alpha * surprise_tilde)
        
        # Clamp to valid range
        phi_t = np.clip(phi_t, self.phi_min, 1.0)
        
        self.metrics['surprise_tilde'] = surprise_tilde
        
        return phi_t
    
    def compute_jitter(self, psi_t: float) -> float:
        """
        Compute the Chronos-Jitter (χ_t) for phase-lock breaking.
        
        Args:
            psi_t: Entropy Guard value
        
        Returns:
            jitter_t: Learning rate jitter value
        """
        # jitter = uniform(-scale, +scale) · (1 - Ψ_t)
        # When Ψ_t is small (resonance), jitter is large
        # When Ψ_t is close to 1 (normal), jitter is small
        
        jitter_magnitude = np.random.uniform(-self.jitter_scale, self.jitter_scale)
        jitter_t = jitter_magnitude * (1.0 - psi_t)
        
        return jitter_t
    
    def step(self, loss: float, model: Optional[torch.nn.Module] = None):
        """
        Perform an optimization step with ARS.
        
        Args:
            loss: The current loss value
            model: The model (optional, used for some surprise metrics)
        """
        self.step_count += 1
        
        # 1. Compute surprise
        surprise_t = self.compute_surprise(loss, model)
        self.surprise_history.append(surprise_t)
        self.metrics['surprise_t'] = surprise_t
        
        # 2. Compute Entropy Guard (Ψ_t)
        rho_1, psi_t = self.compute_entropy_guard()
        self.metrics['rho_1'] = rho_1
        self.metrics['psi_t'] = psi_t
        
        # 3. Compute Surprise Gate (Φ_t)
        if self.enable_entropy_guard:
            phi_t = self.compute_damping_gate(surprise_t, psi_t)
        else:
            phi_t = 1.0
        self.metrics['phi_t'] = phi_t
        
        # 4. Compute Chronos-Jitter (χ_t)
        if self.enable_jitter:
            jitter_t = self.compute_jitter(psi_t)
        else:
            jitter_t = 0.0
        self.metrics['jitter_t'] = jitter_t
        
        # 5. Apply damping to momentum parameters (for AdamW-like optimizers)
        self._apply_damping(phi_t)
        
        # 6. Apply jitter to learning rate
        self._apply_jitter(jitter_t)
        
        # 7. Perform the actual optimizer step
        self.base_optimizer.step()
        
        # 8. Restore original parameters for scheduler compatibility
        self._restore_parameters()
    
    def _apply_damping(self, phi_t: float):
        """
        Apply damping to the momentum parameters of the base optimizer.
        
        This modifies the beta1 and beta2 parameters of AdamW-like optimizers.
        
        Args:
            phi_t: Damping factor (0.0 to 1.0)
        """
        for group in self.base_optimizer.param_groups:
            if 'betas' in group:
                original_betas = self.original_betas.get(id(group), group['betas'])
                beta1_original, beta2_original = original_betas
                
                # Apply damping: β_eff = β_original · Φ_t
                group['betas'] = (
                    beta1_original * phi_t,
                    beta2_original * phi_t
                )
    
    def _apply_jitter(self, jitter_t: float):
        """
        Apply jitter to the learning rate.
        
        Args:
            jitter_t: Jitter value to apply
        """
        for group in self.base_optimizer.param_groups:
            if 'lr' in group:
                original_lr = group['lr']
                group['lr'] = original_lr * (1.0 + jitter_t)
    
    def _restore_parameters(self):
        """
        Restore original optimizer parameters for scheduler compatibility.
        """
        for group in self.base_optimizer.param_groups:
            # Restore betas
            if 'betas' in group:
                original_betas = self.original_betas.get(id(group), group['betas'])
                group['betas'] = original_betas
            
            # Note: Learning rate is NOT restored here, as it may be modified by the scheduler
            # If you want to restore it, uncomment the line below and store original_lr
            # group['lr'] = original_lr
    
    def zero_grad(self):
        """Zero the gradients of the base optimizer."""
        self.base_optimizer.zero_grad()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the current ARS metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
    
    def state_dict(self):
        """Get the state dictionary of the base optimizer."""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load the state dictionary into the base optimizer."""
        self.base_optimizer.load_state_dict(state_dict)


class ARSOptimizerFactory:
    """
    Factory for creating ARS-wrapped optimizers.
    """
    
    @staticmethod
    def create_ars_adamw(
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.01,
        ars_config: Optional[Dict[str, Any]] = None,
    ) -> ARSOptimizer:
        """
        Create an ARS-wrapped AdamW optimizer.
        
        Args:
            model: The model to optimize
            lr: Learning rate
            betas: Beta parameters for AdamW
            weight_decay: Weight decay
            ars_config: Configuration dictionary for ARS
        
        Returns:
            ARSOptimizer instance
        """
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        
        ars_config = ars_config or {}
        return ARSOptimizer(base_optimizer, **ars_config)
