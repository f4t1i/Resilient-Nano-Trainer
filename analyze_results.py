"""
Analysis and Visualization Script for ARS Experiment Results

This script loads the results from multiple training runs (baseline, +Ψ, +Ψ +χ)
and creates comparative visualizations and quantitative analysis.

Usage:
    python analyze_results.py --runs baseline_run ars_psi_run ars_full_run
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ResultsAnalyzer:
    """Analyze and visualize training results."""
    
    def __init__(self, runs_dir: str = "runs"):
        """Initialize analyzer."""
        self.runs_dir = Path(runs_dir)
    
    def load_run(self, run_name: str) -> Dict:
        """Load results from a single run."""
        run_path = self.runs_dir / run_name
        
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        # Load results.json
        with open(run_path / "results.json", 'r') as f:
            results = json.load(f)
        
        # Load metrics.json if it exists
        metrics = None
        if (run_path / "metrics.json").exists():
            with open(run_path / "metrics.json", 'r') as f:
                metrics = json.load(f)
        
        return {
            'name': run_name,
            'path': run_path,
            'results': results,
            'metrics': metrics,
        }
    
    def load_multiple_runs(self, run_names: List[str]) -> List[Dict]:
        """Load results from multiple runs."""
        runs = []
        for name in run_names:
            try:
                run = self.load_run(name)
                runs.append(run)
                print(f"✓ Loaded run: {name}")
            except Exception as e:
                print(f"✗ Failed to load run {name}: {e}")
        
        return runs
    
    def compute_statistics(self, run: Dict) -> Dict:
        """Compute statistics for a run."""
        results = run['results']
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        
        stats = {
            'name': run['name'],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'min_train_loss': min(train_losses),
            'min_val_loss': min(val_losses),
            'mean_train_loss': np.mean(train_losses),
            'mean_val_loss': np.mean(val_losses),
            'std_train_loss': np.std(train_losses),
            'std_val_loss': np.std(val_losses),
        }
        
        # Compute metrics specific to ARS runs
        if run['metrics'] is not None:
            metrics = run['metrics']
            phi_values = [m['phi_t'] for m in metrics]
            psi_values = [m['psi_t'] for m in metrics]
            rho_values = [m['rho_1'] for m in metrics]
            
            stats['mean_phi'] = np.mean(phi_values)
            stats['mean_psi'] = np.mean(psi_values)
            stats['mean_rho'] = np.mean(rho_values)
            stats['max_phi'] = max(phi_values)
            stats['min_phi'] = min(phi_values)
        
        return stats
    
    def print_statistics(self, runs: List[Dict]):
        """Print statistics for multiple runs."""
        print("\n" + "=" * 100)
        print("STATISTICS SUMMARY")
        print("=" * 100)
        
        for run in runs:
            stats = self.compute_statistics(run)
            
            print(f"\nRun: {stats['name']}")
            print("-" * 100)
            print(f"  Final Train Loss: {stats['final_train_loss']:.6f}")
            print(f"  Final Val Loss:   {stats['final_val_loss']:.6f}")
            print(f"  Min Train Loss:   {stats['min_train_loss']:.6f}")
            print(f"  Min Val Loss:     {stats['min_val_loss']:.6f}")
            print(f"  Mean Train Loss:  {stats['mean_train_loss']:.6f} (±{stats['std_train_loss']:.6f})")
            print(f"  Mean Val Loss:    {stats['mean_val_loss']:.6f} (±{stats['std_val_loss']:.6f})")
            
            if 'mean_phi' in stats:
                print(f"\n  ARS Metrics:")
                print(f"    Mean Φ_t (Damping):  {stats['mean_phi']:.4f} (range: {stats['min_phi']:.4f} - {stats['max_phi']:.4f})")
                print(f"    Mean Ψ_t (Entropy):  {stats['mean_psi']:.4f}")
                print(f"    Mean ρ₁ (Autocorr):  {stats['mean_rho']:.4f}")
    
    def plot_loss_comparison(self, runs: List[Dict], output_path: Optional[str] = None):
        """Plot training loss comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training loss
        ax = axes[0]
        for run in runs:
            train_losses = run['results']['train_losses']
            ax.plot(train_losses, label=run['name'], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1]
        for run in runs:
            val_losses = run['results']['val_losses']
            eval_interval = run['results']['config']['max_iters'] // len(val_losses)
            steps = [i * eval_interval for i in range(len(val_losses))]
            ax.plot(steps, val_losses, marker='o', label=run['name'], linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to {output_path}")
        
        return fig
    
    def plot_ars_metrics(self, runs: List[Dict], output_path: Optional[str] = None):
        """Plot ARS-specific metrics."""
        # Filter runs that have metrics
        ars_runs = [r for r in runs if r['metrics'] is not None]
        
        if not ars_runs:
            print("No ARS runs found with metrics.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot for each ARS run
        for run in ars_runs:
            metrics = run['metrics']
            steps = [m['step'] for m in metrics]
            
            # Φ_t (Damping Gate)
            ax = axes[0, 0]
            phi_values = [m['phi_t'] for m in metrics]
            ax.plot(steps, phi_values, label=run['name'], linewidth=2, alpha=0.8)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Φ_t', fontsize=11)
            ax.set_title('Damping Gate (Φ_t)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            # Ψ_t (Entropy Guard)
            ax = axes[0, 1]
            psi_values = [m['psi_t'] for m in metrics]
            ax.plot(steps, psi_values, label=run['name'], linewidth=2, alpha=0.8)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Ψ_t', fontsize=11)
            ax.set_title('Entropy Guard (Ψ_t)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # ρ₁ (Autocorrelation)
            ax = axes[1, 0]
            rho_values = [m['rho_1'] for m in metrics]
            ax.plot(steps, rho_values, label=run['name'], linewidth=2, alpha=0.8)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('ρ₁', fontsize=11)
            ax.set_title('Lag-1 Autocorrelation (ρ₁)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1.1, 1.1])
            
            # Loss with Φ_t overlay
            ax1 = axes[1, 1]
            loss_values = [m['loss'] for m in metrics]
            ax1.plot(steps, loss_values, label='Loss', linewidth=2, color='blue', alpha=0.8)
            ax1.set_xlabel('Step', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)
            
            # Overlay Φ_t
            ax2 = ax1.twinx()
            ax2.plot(steps, phi_values, label='Φ_t', linewidth=2, color='red', alpha=0.6, linestyle='--')
            ax2.set_ylabel('Φ_t', fontsize=11, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim([0, 1.1])
            
            ax1.set_title('Loss with Damping Gate Overlay', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=10)
            ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {output_path}")
        
        return fig
    
    def plot_weight_norm_analysis(self, runs: List[Dict], output_path: Optional[str] = None):
        """
        Plot weight norm analysis (if available from metrics).
        
        Note: This requires logging weight norms during training.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for run in runs:
            if run['metrics'] is not None:
                metrics = run['metrics']
                steps = [m['step'] for m in metrics]
                
                # Try to extract weight norm if available
                # This would need to be added to the training script
                # For now, we'll skip this
        
        return fig
    
    def generate_report(self, runs: List[Dict], output_dir: str = "analysis"):
        """Generate a complete analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating analysis report in {output_dir}...")
        
        # Print statistics
        self.print_statistics(runs)
        
        # Plot loss comparison
        loss_fig = self.plot_loss_comparison(runs, output_dir / "loss_comparison.png")
        
        # Plot ARS metrics
        metrics_fig = self.plot_ars_metrics(runs, output_dir / "ars_metrics.png")
        
        # Save summary to text file
        with open(output_dir / "summary.txt", 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("ADAPTIVE RESONANCE SUPPRESSION (ARS) EXPERIMENT RESULTS\n")
            f.write("=" * 100 + "\n\n")
            
            for run in runs:
                stats = self.compute_statistics(run)
                f.write(f"Run: {stats['name']}\n")
                f.write("-" * 100 + "\n")
                f.write(f"  Final Train Loss: {stats['final_train_loss']:.6f}\n")
                f.write(f"  Final Val Loss:   {stats['final_val_loss']:.6f}\n")
                f.write(f"  Min Train Loss:   {stats['min_train_loss']:.6f}\n")
                f.write(f"  Min Val Loss:     {stats['min_val_loss']:.6f}\n")
                f.write(f"  Mean Train Loss:  {stats['mean_train_loss']:.6f} (±{stats['std_train_loss']:.6f})\n")
                f.write(f"  Mean Val Loss:    {stats['mean_val_loss']:.6f} (±{stats['std_val_loss']:.6f})\n")
                
                if 'mean_phi' in stats:
                    f.write(f"\n  ARS Metrics:\n")
                    f.write(f"    Mean Φ_t (Damping):  {stats['mean_phi']:.4f}\n")
                    f.write(f"    Mean Ψ_t (Entropy):  {stats['mean_psi']:.4f}\n")
                    f.write(f"    Mean ρ₁ (Autocorr):  {stats['mean_rho']:.4f}\n")
                
                f.write("\n")
        
        print(f"✓ Report saved to {output_dir / 'summary.txt'}")
        print(f"✓ Plots saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize ARS experiment results")
    parser.add_argument("--runs", type=str, nargs='+', required=True,
                        help="Names of run directories to analyze")
    parser.add_argument("--output-dir", type=str, default="analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--runs-dir", type=str, default="runs",
                        help="Directory containing run folders")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResultsAnalyzer(runs_dir=args.runs_dir)
    
    # Load runs
    print("Loading runs...")
    runs = analyzer.load_multiple_runs(args.runs)
    
    if not runs:
        print("No runs loaded. Exiting.")
        exit(1)
    
    # Generate report
    analyzer.generate_report(runs, output_dir=args.output_dir)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)
