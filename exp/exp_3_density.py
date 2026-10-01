"""
Experiment 3: Short-maturity density expansion validation for basket returns.

Validates q_theta(x) = phi(x + sigma_{0,theta}/2) * [1 + c_theta H_3(x + sigma_{0,theta}/2)]
and the convergence of scaled Y_0^theta to its asymptotic value.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.coefficients import baseline_A_config, baseline_B_config
from src.monte_carlo import MATURITY_LADDER, run_density_experiment

_phi = norm.pdf

CONFIGS = {"A": baseline_A_config, "B": baseline_B_config}
H_VALUES = [0.1, 0.3]

# Maturities selected for the density subplot
DENSITY_PLOT_THETAS = [2.0**-7, 2.0**-9, 2.0**-12]


def run_density_for_scenario(
    config_name: str,
    H: float,
    M: int = 256,
    N_half: int = 2000,
    n_batches: int = 20,
    base_seed: int = 200,
    fast: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the density experiment across all maturities for one (config, H) scenario.

    Parameters
    ----------
    config_name : "A" or "B"
    H : Hurst exponent
    M, N_half, n_batches, base_seed : simulation parameters

    Returns
    -------
    df : DataFrame with scalar summary columns per maturity
    density_data : dict keyed by theta, each value holding x_grid / q_theory /
                   q_empirical / bin_centers / hist_density arrays
    """
    if fast:
        N_half = min(N_half, 500)
        n_batches = min(n_batches, 5)

    config = CONFIGS[config_name]()
    rows = []
    density_data: dict = {}

    maturities = MATURITY_LADDER[::2] if fast else MATURITY_LADDER
    for theta in maturities:
        print(f"    theta={theta:.6f}")
        res = run_density_experiment(config, H, theta, None, M, N_half, n_batches, base_seed)

        rows.append(
            {
                "theta": res["theta"],
                "H": res["H"],
                "sigma_0_theta": res["sigma_0_theta"],
                "c_theta": res["c_theta"],
                "Y0": res["Y0"],
                "Y0_theory": res["Y0_theory"],
                "digital_mean": res["digital_mean"],
                "psi_A": res["psi_A"],
                "psi_B": res["psi_B"],
            }
        )
        density_data[float(theta)] = {
            "x_grid": res["x_grid"],
            "q_theory": res["q_theory"],
            "q_empirical": res["q_empirical"],
            "bin_centers": res["bin_centers"],
            "hist_density": res["hist_density"],
        }

    df = pd.DataFrame(rows)
    return df, density_data


def plot_density(
    all_results: tuple[pd.DataFrame, dict],
    config_name: str,
    H: float,
    save_path: Path,
) -> None:
    """
    Two-panel plot:
      Left  – empirical histogram density vs theoretical q_theta(x) for three maturities.
      Right – scaled Y_0 / (theta^H + sqrt(theta)) vs theta with asymptotic theory line.

    Parameters
    ----------
    all_results : (df, density_data) tuple returned by run_density_for_scenario
    config_name : "A" or "B"
    H : Hurst exponent
    save_path : output file path
    """
    df, density_data = all_results

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ------------------------------------------------------------------
    # Left panel: density comparison for three selected maturities
    # ------------------------------------------------------------------
    ax = axes[0]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for theta_sel, color in zip(DENSITY_PLOT_THETAS, colors):
        # Locate the closest key stored in density_data
        key = min(density_data.keys(), key=lambda t: abs(t - theta_sel))
        data = density_data[key]

        bc = data["bin_centers"]
        hd = data["hist_density"]
        xg = data["x_grid"]
        qt = data["q_theory"]

        m = -int(round(np.log2(key)))
        bw = (bc[1] - bc[0]) if len(bc) > 1 else 0.5
        ax.bar(bc, hd, width=bw, alpha=0.30, color=color, label=f"Empirical θ=2⁻{m}")

        mask = ~np.isnan(qt)
        ax.plot(xg[mask], qt[mask], color=color, lw=2.0, label=f"Theory θ=2⁻{m}")

    ax.set_xlabel("Normalised log-return x")
    ax.set_ylabel("Density")
    ax.set_title(f"Short-Maturity Density  |  Baseline {config_name}, H={H}")
    ax.legend(fontsize=7, ncol=2)

    # ------------------------------------------------------------------
    # Right panel: Y0 / (r_theta + sqrt_theta) vs theta
    # ------------------------------------------------------------------
    ax2 = axes[1]

    thetas = df["theta"].values
    r_theta = thetas**H
    sqrt_theta = np.sqrt(thetas)
    scale = r_theta + sqrt_theta

    y0_scaled = df["Y0"].values / scale

    ax2.semilogx(thetas, y0_scaled, "o-", color="steelblue", label="MC  Y₀ (scaled)")

    # Approximate horizontal theory level: phi(0)*[psi_A/(2H+3) + psi_B/4]
    psi_A = float(df["psi_A"].iloc[0])
    psi_B = float(df["psi_B"].iloc[0])
    theory_level = _phi(0.0) * (psi_A / (2.0 * H + 3.0) + psi_B / 4.0)
    ax2.axhline(
        theory_level,
        color="crimson",
        ls="--",
        lw=1.5,
        label=f"φ(0)·[ψ_A/(2H+3)+ψ_B/4] ≈ {theory_level:.4f}",
    )

    ax2.set_xlabel("θ  (maturity)")
    ax2.set_ylabel("Y₀ / (θ^H + √θ)")
    ax2.set_title(f"Scaled Y₀ Convergence  |  Baseline {config_name}, H={H}")
    ax2.legend(fontsize=9)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def main(fast: bool = False) -> dict:
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    H_list = [0.1] if fast else H_VALUES
    n_scenarios = 0

    for config_name in ["A", "B"]:
        for H in H_list:
            print(f"\nBaseline {config_name}, H={H}")
            results = run_density_for_scenario(config_name, H, fast=fast)
            df = results[0]
            n_scenarios += 1

            csv_path = RESULTS_DIR / f"density_baseline{config_name}_H{H}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

            png_path = RESULTS_DIR / f"density_baseline{config_name}_H{H}.png"
            plot_density(results, config_name, H, png_path)
            print(f"  Saved {png_path}")

    summary_path = str(RESULTS_DIR / "density_summary.csv")
    return {"status": "done", "n_scenarios": n_scenarios, "summary_path": summary_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
