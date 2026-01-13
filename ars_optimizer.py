"""
Adaptive Resonance Suppression (ARS) Optimizer
Wraps any PyTorch optimizer with stability mechanisms
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math

class ARSOptimizer:
    """
    ARS Optimizer Wrapper with:
    - Entropy Guard (Ψ_t): Detects periodicity via lag-1 autocorrelation
    - Surprise Gate (Φ_t): Adaptive momentum damping
    - Chronos-Jitter (χ_t): Phase-lock breaking noise
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        alpha: float = 2.0,
        phi_min: float = 0.1,
        jitter_scale: float = 0.01,
        window_size: int = 50,
        rho_threshold: float = 0.7
    ):
        self.optimizer = base_optimizer
        self.alpha = alpha
        self.phi_min = phi_min
        self.jitter_scale = jitter_scale
        self.window_size = window_size
        self.rho_threshold = rho_threshold
        
        # History buffers
        self.surprise_history = []
        self.loss_history = []
        
        # Metrics
        self.phi_t = 1.0  # Surprise gate
        self.psi_t = 1.0  # Entropy guard
        self.rho_1 = 0.0  # Lag-1 autocorrelation
        
    def compute_surprise(self, loss: float) -> float:
        """Compute surprise as deviation from recent mean"""
        if len(self.loss_history) < 2:
            return 0.0
        
        recent_mean = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
        return abs(loss - recent_mean)
    
    def compute_autocorrelation(self) -> float:
        """Compute lag-1 autocorrelation of surprise"""
        if len(self.surprise_history) < self.window_size:
            return 0.0
        
        recent = self.surprise_history[-self.window_size:]
        mean = sum(recent) / len(recent)
        
        numerator = sum((recent[i] - mean) * (recent[i+1] - mean) 
                       for i in range(len(recent)-1))
        denominator = sum((x - mean)**2 for x in recent)
        
        if denominator < 1e-8:
            return 0.0
        
        return numerator / denominator
    
    def compute_entropy_guard(self) -> float:
        """Ψ_t: Detects resonance via autocorrelation"""
        if abs(self.rho_1) > self.rho_threshold:
            # High autocorrelation = potential resonance
            return max(0.1, 1.0 - abs(self.rho_1))
        return 1.0
    
    def compute_surprise_gate(self, surprise: float) -> float:
        """Φ_t: Adaptive damping based on adjusted surprise"""
        adjusted_surprise = surprise / self.psi_t
        gate = 1.0 - math.tanh(self.alpha * adjusted_surprise)
        return max(self.phi_min, gate)
    
    def apply_chronos_jitter(self):
        """χ_t: Add noise to break phase-lock"""
        if self.psi_t < 0.5:  # Only when resonance detected
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * self.jitter_scale
                        p.grad.add_(noise)
    
    def step(self, loss: float):
        """Perform optimization step with ARS"""
        # Compute surprise
        surprise = self.compute_surprise(loss)
        self.surprise_history.append(surprise)
        self.loss_history.append(loss)
        
        # Trim histories
        if len(self.surprise_history) > self.window_size * 2:
            self.surprise_history = self.surprise_history[-self.window_size:]
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
        
        # Compute metrics
        self.rho_1 = self.compute_autocorrelation()
        self.psi_t = self.compute_entropy_guard()
        self.phi_t = self.compute_surprise_gate(surprise)
        
        # Apply jitter if needed
        self.apply_chronos_jitter()
        
        # Scale gradients by surprise gate
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.mul_(self.phi_t)
        
        # Perform base optimizer step
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'surprise_history': self.surprise_history,
            'loss_history': self.loss_history,
            'phi_t': self.phi_t,
            'psi_t': self.psi_t,
            'rho_1': self.rho_1
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.surprise_history = state_dict['surprise_history']
        self.loss_history = state_dict['loss_history']
        self.phi_t = state_dict['phi_t']
        self.psi_t = state_dict['psi_t']
        self.rho_1 = state_dict['rho_1']
