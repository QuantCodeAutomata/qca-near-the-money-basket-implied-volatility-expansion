"""
Experiment 1: Near-the-money implied volatility smile expansion for basket options.

Validates the asymptotic expansion:
    sigma(z, theta) ≈ sqrt(bar_v(theta)) + c_theta * z
where c_theta = (psi_A / (2H+3)) * r(theta) + (psi_B / 4) * sqrt(theta)
and r(theta) = theta^H.

Runs over a maturity ladder and a z-grid, comparing MC implied vols to the
full expansion and its A-only / B-only components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coefficients import (
    baseline_A_config,
    baseline_B_config,
    compute_scenario_coefficients,
)
from src.monte_carlo import run_smile_experiment

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path('results')
MATURITY_LADDER = [2**-m for m in range(6, 13)]   # [1/64, ..., 1/4096]
Z_GRID = [-2, -1, -0.5, 0, 0.5, 1, 2]


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_smile_for_scenario(
    config_name: str,
    H: float,
    M: int = 256,
    N_half: int = 2000,
    n_batches: int = 20,
    base_seed: int = 0,
    fast: bool = False,
) -> pd.DataFrame:
    """
    Run the near-the-money smile experiment for one baseline / H combination.

    Parameters
    ----------
    config_name : 'A' or 'B'
    H : float — Hurst parameter
    M : int — time steps per maturity
    N_half : int — antithetic path pairs per batch
    n_batches : int — independent batches
    base_seed : int — base RNG seed

    Returns
    -------
    df : DataFrame with columns
        theta, z, sigma_MC, sigma_full, sigma_A_only, sigma_B_only,
        abs_error, scaled_residual, se_MC
    """
    if fast:
        N_half = min(N_half, 500)
        n_batches = min(n_batches, 5)

    config = baseline_A_config() if config_name == 'A' else baseline_B_config()
    coeffs = compute_scenario_coefficients(config, H)
    psi_A = coeffs['psi_A']
    psi_B = coeffs['psi_B']
    v0 = coeffs['v0']

    print(f"  Baseline {config_name} | H={H:.1f} | psi_A={psi_A:.4f} "
          f"psi_B={psi_B:.4f} v0={v0:.4f}")

    z_arr = np.array(Z_GRID)
    maturities = MATURITY_LADDER[::2] if fast else MATURITY_LADDER  # every-other for fast
    frames = []

    for theta in maturities:
        df_theta = run_smile_experiment(
            config, H, theta, z_arr, M, N_half, n_batches, base_seed
        )
        df_theta.insert(0, 'theta', theta)
        frames.append(df_theta)

    df = pd.concat(frames, ignore_index=True)
    return df[['theta', 'z', 'sigma_MC', 'sigma_full', 'sigma_A_only',
               'sigma_B_only', 'abs_error', 'scaled_residual', 'se_MC']]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_smile_slices(
    df: pd.DataFrame,
    config_name: str,
    H: float,
    save_path: Path,
) -> None:
    """
    Two-panel figure:
      Left  — sigma_MC and sigma_full vs z for each theta (colour-coded).
      Right — |scaled_residual| at z=0 vs theta on a log-log scale.

    Saves the figure to save_path as a PNG.
    """
    thetas = sorted(df['theta'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: smile slices ---
    for theta, color in zip(thetas, colors):
        sub = df[df['theta'] == theta].sort_values('z')
        lbl = f'θ={theta:.5f}'
        ax1.plot(sub['z'], sub['sigma_MC'], 'o', color=color,
                 markersize=4, label=f'MC  {lbl}')
        ax1.plot(sub['z'], sub['sigma_full'], '--', color=color,
                 linewidth=1.2, label=f'Theory {lbl}')

    ax1.set_xlabel('z  (normalised log-moneyness)', fontsize=11)
    ax1.set_ylabel('Implied Volatility σ(z, θ)', fontsize=11)
    ax1.set_title(f'Smile Slices — Baseline {config_name}, H = {H}', fontsize=12)
    ax1.legend(fontsize=5, ncol=2, loc='best')
    ax1.grid(True, alpha=0.3)

    # --- Right: z=0 scaled residual vs theta ---
    df0 = df[df['z'] == 0.0].sort_values('theta').copy()
    df0 = df0.dropna(subset=['scaled_residual'])
    ax2.plot(df0['theta'], df0['scaled_residual'].abs(), 'o-',
             color='steelblue', linewidth=1.5, markersize=5,
             label='|scaled residual| at z=0')
    ax2.set_xlabel('θ (maturity)', fontsize=11)
    ax2.set_ylabel('|scaled residual|  / (r(θ) + √θ)', fontsize=11)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'Convergence at z = 0', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        f'Near-the-Money Smile Expansion — Baseline {config_name}, H = {H}',
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    H_list = [0.1, 0.5] if fast else [0.1, 0.3, 0.5]

    for baseline in ['A', 'B']:
        for H in H_list:
            print(f"\n{'='*60}")
            print(f"Baseline {baseline}  |  H = {H}")
            print('='*60)

            df = run_smile_for_scenario(baseline, H, fast=fast)

            csv_path = RESULTS_DIR / f'smile_baseline{baseline}_H{H}.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV  → {csv_path}")

            png_path = RESULTS_DIR / f'smile_baseline{baseline}_H{H}.png'
            plot_smile_slices(df, baseline, H, png_path)
            print(f"  Saved plot → {png_path}")

            # Collect per-theta summary stats (mean |abs_error| across z)
            for theta in sorted(df['theta'].unique()):
                sub = df[df['theta'] == theta]
                summary_rows.append({
                    'baseline': baseline,
                    'H': H,
                    'theta': round(theta, 8),
                    'mean_abs_error': float(sub['abs_error'].abs().mean()),
                    'max_abs_error': float(sub['abs_error'].abs().max()),
                    'mean_scaled_residual': float(sub['scaled_residual'].abs().mean()),
                    'sigma_MC_atm': float(
                        sub.loc[sub['z'] == 0.0, 'sigma_MC'].values[0]
                        if (sub['z'] == 0.0).any() else np.nan
                    ),
                    'sigma_full_atm': float(
                        sub.loc[sub['z'] == 0.0, 'sigma_full'].values[0]
                        if (sub['z'] == 0.0).any() else np.nan
                    ),
                })

    summary = pd.DataFrame(summary_rows)
    print('\n' + '='*60)
    print('SUMMARY  (mean |abs_error| across z, per maturity)')
    print('='*60)
    tbl = summary.pivot_table(
        index=['baseline', 'H', 'theta'],
        values=['mean_abs_error', 'max_abs_error', 'mean_scaled_residual'],
        aggfunc='first',
    )
    print(tbl.to_string(float_format='{:.6f}'.format))

    summary_path = RESULTS_DIR / 'smile_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved → {summary_path}")

    return {"status": "done", "n_scenarios": len(summary_rows), "summary_path": str(summary_path)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    main(fast=args.fast)
