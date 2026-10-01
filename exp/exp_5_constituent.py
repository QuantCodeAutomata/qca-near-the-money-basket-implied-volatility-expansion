"""
Experiment 5: Single-asset constituent-level asymptotic validation.

For each constituent i, validates the short-maturity implied-vol expansion:
  sigma_asym_i(z, theta) = sqrt(V0[i]) + psi_ii / (2H+3) * theta^H * z
against Monte Carlo put-implied volatilities, using direct simulation of
constituent terminal prices from simulate_model_paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.blackscholes import implied_vol_from_price, put_price_from_mc
from src.coefficients import baseline_A_config, baseline_B_config
from src.monte_carlo import MATURITY_LADDER, Z_GRID
from src.simulation import simulate_model_paths

CONFIGS = {"A": baseline_A_config, "B": baseline_B_config}
H_VALUES = [0.1, 0.3]

# Maturities used in the smile-slice subplot
SMILE_PLOT_THETAS = [2.0**-6, 2.0**-9, 2.0**-12]


def compute_constituent_theory(config: dict, H: float) -> dict:
    """
    Compute per-constituent asymptotic theory coefficients.

    For each constituent i:
      gamma_ii  = beta[i] * rho_loadings[i, 0]          (self-coupling)
      psi_ii    = 2 * gamma_ii / (2H + 1)
      bar_vi    = V0[i]                                   (E[V_t^i] = V_0^i exactly)
      sigma_asym_i(z, theta) = sqrt(V0[i]) + psi_ii/(2H+3) * theta^H * z

    Parameters
    ----------
    config : baseline configuration dict
    H : Hurst exponent

    Returns
    -------
    theory : dict keyed by constituent index i, each value a dict with
             gamma_ii, psi_ii, bar_vi, sigma_asym (callable)
    """
    beta = np.asarray(config["beta"], dtype=float)
    rho_loadings = np.asarray(config["rho_loadings"], dtype=float)
    V0 = np.asarray(config["V0"], dtype=float)
    n = config["n"]

    theory: dict = {}
    for i in range(n):
        gamma_ii = beta[i] * rho_loadings[i, 0]
        psi_ii = 2.0 * gamma_ii / (2.0 * H + 1.0)
        bar_vi = V0[i]

        # Capture loop variables explicitly via default arguments
        def sigma_asym(z, theta, _psi=psi_ii, _v0i=V0[i], _H=H):
            return np.sqrt(_v0i) + (_psi / (2.0 * _H + 3.0)) * theta**_H * z

        theory[i] = {
            "gamma_ii": gamma_ii,
            "psi_ii": psi_ii,
            "bar_vi": bar_vi,
            "sigma_asym": sigma_asym,
        }
    return theory


def run_constituent_for_scenario(
    config_name: str,
    H: float,
    M: int = 256,
    N_half: int = 2000,
    n_batches: int = 20,
    base_seed: int = 400,
    fast: bool = False,
) -> pd.DataFrame:
    """
    Run constituent-level implied-vol experiment across all maturities.

    For each (theta, constituent i, moneyness z):
      - Prices put E[(K_i - S_theta^i)_+] with K_i = S0[i]*exp(sqrt(theta)*z)
        by averaging over n_batches independent simulations.
      - Inverts to MC implied vol sigma_MC_i.
      - Compares to theory sigma_asym_i(z, theta).

    NaN implied vols are stored as NaN in the output (not silently dropped).

    Parameters
    ----------
    config_name : "A" or "B"
    H : Hurst exponent
    M, N_half, n_batches, base_seed : simulation parameters

    Returns
    -------
    df : DataFrame with one row per (theta, constituent, z) triple, columns:
         config, H, theta, constituent, z, S0_i, K_i,
         sigma_MC, sigma_asym, residual, scaled_residual, psi_ii, r_theta
    """
    if fast:
        N_half = min(N_half, 500)
        n_batches = min(n_batches, 5)

    config = CONFIGS[config_name]()
    S0 = np.asarray(config["S0"], dtype=float)
    n = config["n"]
    theory = compute_constituent_theory(config, H)

    rows = []

    maturities = MATURITY_LADDER[::2] if fast else MATURITY_LADDER
    for theta in maturities:
        print(f"    theta={theta:.6f}")
        r_theta = float(theta**H)
        sqrt_theta = float(np.sqrt(theta))

        # Accumulate put prices over batches: shape (n, len(Z_GRID))
        put_prices_acc = np.zeros((n, len(Z_GRID)))

        for b in range(n_batches):
            result = simulate_model_paths(
                config, H, theta, M, N_half, seed=base_seed + b
            )
            S_terminal = result["S_terminal"]  # (n, N)

            for i in range(n):
                for iz, z in enumerate(Z_GRID):
                    K_i = S0[i] * np.exp(sqrt_theta * z)
                    p, _ = put_price_from_mc(S_terminal[i, :], K_i)
                    put_prices_acc[i, iz] += p

        put_prices_avg = put_prices_acc / n_batches

        # Invert average prices and build output rows
        for i in range(n):
            for iz, z in enumerate(Z_GRID):
                K_i = float(S0[i] * np.exp(sqrt_theta * z))
                sigma_MC = implied_vol_from_price(
                    put_prices_avg[i, iz], float(S0[i]), K_i, theta
                )
                sigma_th = theory[i]["sigma_asym"](float(z), theta)

                if not np.isnan(sigma_MC):
                    residual = sigma_MC - sigma_th
                    scaled_residual = residual / (r_theta + sqrt_theta)
                else:
                    residual = np.nan
                    scaled_residual = np.nan

                rows.append(
                    {
                        "config": config_name,
                        "H": H,
                        "theta": float(theta),
                        "constituent": i,
                        "z": float(z),
                        "S0_i": float(S0[i]),
                        "K_i": K_i,
                        "sigma_MC": sigma_MC,
                        "sigma_asym": sigma_th,
                        "residual": residual,
                        "scaled_residual": scaled_residual,
                        "psi_ii": theory[i]["psi_ii"],
                        "r_theta": r_theta,
                    }
                )

    return pd.DataFrame(rows)


def plot_constituent(
    df: pd.DataFrame, config_name: str, H: float, save_path: Path
) -> None:
    """
    Two-panel plot:
      Left  – constituent smile slices (sigma vs z) for three maturities,
               MC markers and theory lines per constituent.
      Right – scaled residuals (sigma_MC - sigma_asym)/(theta^H+sqrt(theta))
               vs theta at z=0 for each constituent.

    Parameters
    ----------
    df : DataFrame returned by run_constituent_for_scenario
    config_name : "A" or "B"
    H : Hurst exponent
    save_path : output file path
    """
    n_constituents = df["constituent"].nunique()
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_constituents))
    linestyles_theta = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ------------------------------------------------------------------
    # Left panel: constituent smile slices
    # ------------------------------------------------------------------
    ax = axes[0]

    for i in range(n_constituents):
        df_i = df[df["constituent"] == i]
        for j, theta_sel in enumerate(SMILE_PLOT_THETAS):
            key = min(df_i["theta"].unique(), key=lambda t: abs(t - theta_sel))
            df_it = df_i[df_i["theta"] == key].sort_values("z")
            z_vals = df_it["z"].values
            sigma_mc = df_it["sigma_MC"].values
            sigma_th = df_it["sigma_asym"].values
            ls = linestyles_theta[j % len(linestyles_theta)]
            m = -int(round(np.log2(key)))

            mask = ~np.isnan(sigma_mc)
            ax.plot(
                z_vals[mask],
                sigma_mc[mask],
                ls=ls,
                marker="o",
                markersize=4,
                color=colors[i],
                label=f"i={i+1}, θ=2⁻{m} MC",
                alpha=0.85,
            )
            ax.plot(
                z_vals,
                sigma_th,
                ls="--",
                color=colors[i],
                lw=1.2,
                alpha=0.5,
                label=f"i={i+1}, θ=2⁻{m} Theory",
            )

    ax.set_xlabel("Moneyness z")
    ax.set_ylabel("Implied volatility σ")
    ax.set_title(f"Constituent Smiles  |  Baseline {config_name}, H={H}")
    ax.legend(fontsize=6, ncol=2)

    # ------------------------------------------------------------------
    # Right panel: scaled residuals at z=0 vs theta
    # ------------------------------------------------------------------
    ax2 = axes[1]

    for i in range(n_constituents):
        df_iz0 = (
            df[(df["constituent"] == i) & (df["z"] == 0.0)]
            .sort_values("theta")
        )
        thetas_i = df_iz0["theta"].values
        sr = df_iz0["scaled_residual"].values
        mask = ~np.isnan(sr)
        ax2.semilogx(
            thetas_i[mask],
            sr[mask],
            "o-",
            color=colors[i],
            label=f"Constituent {i+1}",
        )

    ax2.axhline(0.0, color="gray", ls="--", lw=1.0, label="Zero")
    ax2.set_xlabel("θ  (maturity)")
    ax2.set_ylabel("(σ_MC − σ_asym) / (θ^H + √θ)")
    ax2.set_title(f"Scaled Residuals (z=0)  |  Baseline {config_name}, H={H}")
    ax2.legend(fontsize=8)
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
            df = run_constituent_for_scenario(config_name, H, fast=fast)
            n_scenarios += 1

            csv_path = RESULTS_DIR / f"constituent_baseline{config_name}_H{H}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

            png_path = RESULTS_DIR / f"constituent_baseline{config_name}_H{H}.png"
            plot_constituent(df, config_name, H, png_path)
            print(f"  Saved {png_path}")

    return {"status": "done", "n_scenarios": n_scenarios}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
