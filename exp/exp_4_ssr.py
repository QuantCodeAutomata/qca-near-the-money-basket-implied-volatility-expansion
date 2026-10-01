"""
Experiment 4: SSR asymptotics validation.

Validates that SSR_0(theta) = A_0^theta / Y_0^theta → H + 3/2 as theta → 0,
and checks convergence of the scaled Malliavin numerator A_0 and digital
denominator Y_0 to their theoretical short-maturity limits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coefficients import baseline_A_config, baseline_B_config
from src.monte_carlo import MATURITY_LADDER, run_ssr_experiment

CONFIGS = {"A": baseline_A_config, "B": baseline_B_config}
H_VALUES = [0.1, 0.3]


def run_ssr_for_scenario(
    config_name: str,
    H: float,
    M: int = 256,
    N_half: int = 2000,
    n_batches: int = 20,
    base_seed: int = 300,
    fast: bool = False,
) -> pd.DataFrame:
    """
    Run the SSR experiment across all maturities for one (config, H) scenario.

    Parameters
    ----------
    config_name : "A" or "B"
    H : Hurst exponent
    M, N_half, n_batches, base_seed : simulation parameters

    Returns
    -------
    df : DataFrame with columns:
         theta, H, A0, A0_se, Y0, Y0_se, SSR, SSR_target,
         A0_theory, Y0_theory, psi_A, psi_B, r_theta
    """
    if fast:
        N_half = min(N_half, 500)
        n_batches = min(n_batches, 5)

    config = CONFIGS[config_name]()
    rows = []

    maturities = MATURITY_LADDER[::2] if fast else MATURITY_LADDER
    for theta in maturities:
        print(f"    theta={theta:.6f}")
        res = run_ssr_experiment(config, H, theta, M, N_half, n_batches, base_seed)

        if "error" in res:
            print(f"    Warning – theta={theta:.6f}: {res['error']}")
            continue

        rows.append(
            {
                "theta": res["theta"],
                "H": res["H"],
                "A0": res["A0"],
                "A0_se": res["A0_se"],
                "Y0": res["Y0"],
                "Y0_se": res["Y0_se"],
                "SSR": res["SSR"],
                "SSR_target": res["SSR_target"],
                "A0_theory": res["A0_theory"],
                "Y0_theory": res["Y0_theory"],
                "psi_A": res["psi_A"],
                "psi_B": res["psi_B"],
                "r_theta": res["r_theta"],
            }
        )

    return pd.DataFrame(rows)


def plot_ssr(df: pd.DataFrame, config_name: str, H: float, save_path: Path) -> None:
    """
    Two-panel plot:
      Left  – A_0 and Y_0 scaled by (theta^H + sqrt(theta)) vs theta,
               together with their theory lines.
      Right – SSR_0(theta) vs theta with horizontal target at H + 3/2.

    Parameters
    ----------
    df : DataFrame returned by run_ssr_for_scenario
    config_name : "A" or "B"
    H : Hurst exponent
    save_path : output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    thetas = df["theta"].values
    r_theta = df["r_theta"].values
    sqrt_theta = np.sqrt(thetas)
    scale = r_theta + sqrt_theta

    # ------------------------------------------------------------------
    # Left panel: scaled A0 and Y0
    # ------------------------------------------------------------------
    ax = axes[0]

    A0_scaled = df["A0"].values / scale
    Y0_scaled = df["Y0"].values / scale
    A0_theory_scaled = df["A0_theory"].values / scale
    Y0_theory_scaled = df["Y0_theory"].values / scale

    ax.semilogx(thetas, A0_scaled, "o-", color="tab:blue", label="MC  A₀ (scaled)")
    ax.semilogx(
        thetas, A0_theory_scaled, "--", color="tab:blue", lw=1.5, label="Theory  A₀"
    )
    ax.semilogx(thetas, Y0_scaled, "s-", color="tab:orange", label="MC  Y₀ (scaled)")
    ax.semilogx(
        thetas, Y0_theory_scaled, "--", color="tab:orange", lw=1.5, label="Theory  Y₀"
    )

    ax.set_xlabel("θ  (maturity)")
    ax.set_ylabel("Value / (θ^H + √θ)")
    ax.set_title(f"Scaled A₀ and Y₀  |  Baseline {config_name}, H={H}")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # ------------------------------------------------------------------
    # Right panel: SSR vs theta
    # ------------------------------------------------------------------
    ax2 = axes[1]

    SSR_target = H + 1.5
    SSR_vals = df["SSR"].values
    mask = ~np.isnan(SSR_vals)

    ax2.semilogx(
        thetas[mask], SSR_vals[mask], "o-", color="tab:green", label="MC  SSR₀(θ)"
    )
    ax2.axhline(
        SSR_target,
        color="crimson",
        ls="--",
        lw=1.5,
        label=f"Target  H+3/2 = {SSR_target:.3f}",
    )

    ax2.set_xlabel("θ  (maturity)")
    ax2.set_ylabel("SSR₀(θ)")
    ax2.set_title(f"SSR Asymptotics  |  Baseline {config_name}, H={H}")
    ax2.legend(fontsize=9)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _ssr_pass_fail(df: pd.DataFrame, H: float) -> tuple[bool, float, float, float]:
    """
    Check whether SSR at the smallest theta lies within 2 SE of the H+3/2 target.

    Returns (passed, SSR_val, SSR_target, SSR_se).
    """
    row = df.loc[df["theta"].idxmin()]
    SSR_val = float(row["SSR"])
    SSR_target = H + 1.5

    A0 = float(row["A0"])
    Y0 = float(row["Y0"])
    A0_se = float(row["A0_se"])
    Y0_se = float(row["Y0_se"])

    if abs(Y0) > 1e-12:
        # Delta-method error propagation for ratio A0/Y0
        SSR_se = float(
            np.sqrt((A0_se / Y0) ** 2 + (A0 * Y0_se / Y0**2) ** 2)
        )
    else:
        SSR_se = np.nan

    if np.isnan(SSR_se) or np.isnan(SSR_val):
        passed = False
    else:
        passed = abs(SSR_val - SSR_target) <= 2.0 * SSR_se

    return passed, SSR_val, SSR_target, SSR_se


def main(fast: bool = False) -> dict:
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    H_list = [0.1] if fast else H_VALUES
    n_scenarios = 0
    pass_count = 0
    fail_count = 0

    for config_name in ["A", "B"]:
        for H in H_list:
            print(f"\nBaseline {config_name}, H={H}")
            df = run_ssr_for_scenario(config_name, H, fast=fast)
            n_scenarios += 1

            if not df.empty:
                passed, SSR_val, SSR_target, SSR_se = _ssr_pass_fail(df, H)
                status = "PASS" if passed else "FAIL"
                if passed:
                    pass_count += 1
                else:
                    fail_count += 1
                print(
                    f"  SSR check at smallest theta: "
                    f"SSR={SSR_val:.4f}, target={SSR_target:.3f}, "
                    f"SE={SSR_se:.4f}  →  {status}"
                )

            csv_path = RESULTS_DIR / f"ssr_baseline{config_name}_H{H}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

            png_path = RESULTS_DIR / f"ssr_baseline{config_name}_H{H}.png"
            plot_ssr(df, config_name, H, png_path)
            print(f"  Saved {png_path}")

    return {
        "status": "done",
        "n_scenarios": n_scenarios,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
