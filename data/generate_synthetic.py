"""
Synthetic data generation for Gaussian Volterra-factor basket model experiments.

All data is generated under the risk-neutral measure Q using closed-form formulas
and Monte Carlo simulation. No external market data is used.

Usage:
    python data/generate_synthetic.py --baseline A --H 0.5 --theta 0.015625 --N 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.coefficients import baseline_A_config, baseline_B_config, compute_scenario_coefficients
from src.simulation import simulate_model_paths


H_VALUES = [0.1, 0.3, 0.5]
MATURITY_LADDER = np.array([2.0 ** (-m) for m in range(6, 13)])


def generate_scenario_summary(baseline: str = "A") -> pd.DataFrame:
    """
    Generate a table of scenario parameters for both baselines and all H values.

    Parameters
    ----------
    baseline : str - "A" or "B" or "both"

    Returns
    -------
    DataFrame with scenario parameters and theoretical coefficients.
    """
    configs = {}
    if baseline in ("A", "both"):
        configs["A"] = baseline_A_config()
    if baseline in ("B", "both"):
        configs["B"] = baseline_B_config()

    rows = []
    for name, cfg in configs.items():
        for H in H_VALUES:
            coeffs = compute_scenario_coefficients(cfg, H)
            row = {
                "baseline": name,
                "H": H,
                "n": cfg["n"],
                "d": cfg["d"],
                "q": cfg["q"],
                "S0_1": cfg["S0"][0],
                "S0_2": cfg["S0"][1],
                "V0_1": cfg["V0"][0],
                "V0_2": cfg["V0"][1],
                "Pi0_1": coeffs["Pi0"][0],
                "Pi0_2": coeffs["Pi0"][1],
                "beta_1": cfg["beta"][0],
                "beta_2": cfg["beta"][1],
                "rho12": float(coeffs["rho"][0, 1]),
                "v0": coeffs["v0"],
                "psi_A": coeffs["psi_A"],
                "psi_B": coeffs["psi_B"],
                "C_perp_is_pd": coeffs["C_perp_is_pd"],
                "C_perp_min_eigenval": float(coeffs["C_perp_eigenvalues"].min()),
                "nondegen_A": coeffs["nondegen_A"],
                "nondegen_AB": coeffs["nondegen_AB"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def generate_analytic_variance_curve(
    config: dict,
    H: float,
    theta_values: np.ndarray | None = None,
    n_grid: int = 100,
) -> pd.DataFrame:
    """
    Generate the deterministic frozen-weight variance curve bar v(theta) vs theta.

    Parameters
    ----------
    config : baseline config dict
    H : float
    theta_values : optional array of maturities
    n_grid : int - number of internal quadrature points

    Returns
    -------
    DataFrame with theta, bar_v, r_theta, c_theta, sqrt_barv columns.
    """
    from src.coefficients import (
        compute_scenario_coefficients,
        compute_bar_v,
        compute_c_theta,
    )

    if theta_values is None:
        theta_values = MATURITY_LADDER

    coeffs = compute_scenario_coefficients(config, H)
    rows = []
    for theta in theta_values:
        bar_v = compute_bar_v(
            coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta
        )
        c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        rows.append({
            "theta": theta,
            "bar_v": bar_v,
            "sqrt_barv": np.sqrt(bar_v),
            "r_theta": theta ** H,
            "c_theta": c_theta,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data summary tables.")
    parser.add_argument("--baseline", choices=["A", "B", "both"], default="both")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Scenario summary table
    df_scenario = generate_scenario_summary(args.baseline)
    scenario_path = out_dir / "scenario_parameters.csv"
    df_scenario.to_csv(scenario_path, index=False)
    print(f"Saved scenario parameters to {scenario_path}")
    print(df_scenario.to_string())

    # Variance curves
    for bname, cfg_fn in [("A", baseline_A_config), ("B", baseline_B_config)]:
        if args.baseline not in (bname, "both"):
            continue
        cfg = cfg_fn()
        for H in H_VALUES:
            df_var = generate_analytic_variance_curve(cfg, H)
            var_path = out_dir / f"variance_curve_baseline{bname}_H{H:.1f}.csv"
            df_var.to_csv(var_path, index=False)
            print(f"Saved variance curve to {var_path}")


if __name__ == "__main__":
    main()
