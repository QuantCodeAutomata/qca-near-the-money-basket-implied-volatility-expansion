"""
Experiment 2: ATM skew asymptotics and digital correction.

Validates the leading-order ATM skew expansion:
    ∂_k σ(0, θ) ~ (psi_A / (2H+3)) * θ^{H-1/2} + psi_B / 4    (θ → 0)

and the digital correction identity:
    Y_0^θ  =  Q(S_θ < S_0) − Φ(√θ · σ_ATM / 2)

scaled by (r(θ) + √θ) where r(θ) = θ^H.

For each baseline / H combination, results are written to CSV and a
two-panel diagnostic plot is saved to the results directory.
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
from scipy.stats import norm as _scipy_norm

from src.coefficients import (
    baseline_A_config,
    baseline_B_config,
    compute_scenario_coefficients,
)
from src.monte_carlo import run_atm_skew_experiment

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path('results')
MATURITY_LADDER = [2**-m for m in range(6, 13)]   # [1/64, ..., 1/4096]


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_atm_skew_for_scenario(
    config_name: str,
    H: float,
    M: int = 256,
    N_half: int = 2000,
    n_batches: int = 20,
    base_seed: int = 100,
    fast: bool = False,
) -> pd.DataFrame:
    """
    Run the ATM skew / digital correction experiment for one baseline / H
    combination, sweeping over the full maturity ladder.

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
        theta, H, sigma_ATM, digital, Y0, Y0_theory,
        skew_digital, skew_fd, skew_theory, psi_A, psi_B,
        r_theta, sqrt_theta
    """
    if fast:
        N_half = min(N_half, 500)
        n_batches = min(n_batches, 5)

    config = baseline_A_config() if config_name == 'A' else baseline_B_config()
    coeffs = compute_scenario_coefficients(config, H)
    print(f"  Baseline {config_name} | H={H:.1f} | psi_A={coeffs['psi_A']:.4f} "
          f"psi_B={coeffs['psi_B']:.4f}")

    _required = [
        'theta', 'H', 'sigma_ATM', 'digital', 'Y0', 'Y0_theory',
        'skew_digital', 'skew_fd', 'skew_theory', 'psi_A', 'psi_B',
    ]

    maturities = MATURITY_LADDER[::2] if fast else MATURITY_LADDER
    rows = []
    for theta in maturities:
        result = run_atm_skew_experiment(
            config, H, theta, M, N_half, n_batches, base_seed
        )
        rows.append({col: result[col] for col in _required})

    df = pd.DataFrame(rows)

    # Add derived columns from theta column and H parameter
    df['r_theta'] = df['theta'] ** H
    df['sqrt_theta'] = np.sqrt(df['theta'])

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_atm_skew(
    df: pd.DataFrame,
    config_name: str,
    H: float,
    save_path: Path,
) -> None:
    """
    Two-panel figure:

      Left  — Scaled ATM skew vs θ:
               • H < 0.5: θ^{1/2−H} × skew_digital (MC) and theory line.
               • H = 0.5: raw skew_digital (MC) and theory line.

      Right — Normalised digital correction Y₀ / (r(θ) + √θ) vs θ
               with a horizontal asymptotic theory line.

    Saves the figure to save_path as a PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    thetas = df['theta'].values

    # ----- Left subplot: scaled ATM skew -----
    if H < 0.5:
        exponent = 0.5 - H
        y_mc = df['theta'] ** exponent * df['skew_digital']
        y_theory = df['theta'] ** exponent * df['skew_theory']
        ylabel_left = f'θ^{{{exponent:.2f}}}  ×  ∂_k σ(0, θ)'
    else:
        # H = 0.5: r(θ) = √θ so the natural scale is already θ^0 = const
        y_mc = df['skew_digital']
        y_theory = df['skew_theory']
        ylabel_left = '∂_k σ(0, θ)  (raw ATM skew)'

    ax1.plot(thetas, y_mc, 'o-', color='steelblue', linewidth=1.4,
             markersize=5, label='MC (digital identity)')
    ax1.plot(thetas, y_theory, '--', color='darkorange', linewidth=1.8,
             label='Theory')
    ax1.set_xscale('log')
    ax1.set_xlabel('θ  (maturity)', fontsize=11)
    ax1.set_ylabel(ylabel_left, fontsize=11)
    ax1.set_title('ATM Skew Convergence', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)

    # ----- Right subplot: Y₀ / (r(θ) + √θ) -----
    phi0 = _scipy_norm.pdf(0.0)   # 1/sqrt(2π)
    psi_A = float(df['psi_A'].iloc[0])
    psi_B = float(df['psi_B'].iloc[0])

    if H < 0.5:
        # Leading term: phi(0) * psi_A / (2H+3)   [r_theta dominates sqrt_theta]
        hline_val = phi0 * psi_A / (2.0 * H + 3.0)
    else:
        # H = 0.5: r_theta = sqrt_theta → 2*sqrt_theta in denominator
        # c_theta = (psi_A+psi_B)/4 * sqrt_theta
        # Y0_theory / (2*sqrt_theta) = phi0 * (psi_A+psi_B) / 8
        hline_val = phi0 * (psi_A + psi_B) / 8.0

    y_right = df['Y0'] / (df['r_theta'] + df['sqrt_theta'])

    ax2.plot(thetas, y_right, 'o-', color='steelblue', linewidth=1.4,
             markersize=5, label='MC')
    ax2.axhline(
        hline_val, linestyle='--', color='darkorange', linewidth=1.8,
        label=f'Theory asymptote = {hline_val:.4f}',
    )
    ax2.set_xscale('log')
    ax2.set_xlabel('θ  (maturity)', fontsize=11)
    ax2.set_ylabel('Y₀ / (r(θ) + √θ)', fontsize=11)
    ax2.set_title('Digital Correction Normalised', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        f'ATM Skew Asymptotics — Baseline {config_name}, H = {H}',
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
            print(f"ATM Skew  |  Baseline {baseline}  |  H = {H}")
            print('='*60)

            df = run_atm_skew_for_scenario(baseline, H, fast=fast)

            csv_path = RESULTS_DIR / f'atm_skew_baseline{baseline}_H{H}.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV  → {csv_path}")

            png_path = RESULTS_DIR / f'atm_skew_baseline{baseline}_H{H}.png'
            plot_atm_skew(df, baseline, H, png_path)
            print(f"  Saved plot → {png_path}")

            # Collect summary statistics
            for _, row in df.iterrows():
                skew_err = (
                    float(row['skew_digital']) - float(row['skew_theory'])
                    if not np.isnan(row['skew_digital']) else np.nan
                )
                summary_rows.append({
                    'baseline': baseline,
                    'H': H,
                    'theta': round(float(row['theta']), 8),
                    'sigma_ATM': float(row['sigma_ATM']),
                    'skew_digital': float(row['skew_digital']),
                    'skew_theory': float(row['skew_theory']),
                    'skew_error': skew_err,
                    'Y0': float(row['Y0']),
                    'Y0_theory': float(row['Y0_theory']),
                    'r_theta': float(row['r_theta']),
                    'sqrt_theta': float(row['sqrt_theta']),
                })

    summary = pd.DataFrame(summary_rows)
    print('\n' + '='*60)
    print('SUMMARY  (ATM skew error vs theory, per maturity)')
    print('='*60)
    tbl = summary[['baseline', 'H', 'theta', 'skew_digital', 'skew_theory', 'skew_error']]
    print(tbl.to_string(index=False, float_format='{:.6f}'.format))

    summary_path = RESULTS_DIR / 'atm_skew_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved → {summary_path}")

    return {"status": "done", "n_scenarios": len(summary_rows), "summary_path": str(summary_path)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    main(fast=args.fast)
