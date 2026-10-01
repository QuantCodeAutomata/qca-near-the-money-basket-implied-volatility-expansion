"""
Monte Carlo engine for basket option pricing and asymptotic validation.

Combines simulation, Black-Scholes inversion, and coefficient comparison
for all experiments. Uses independent batch estimation with antithetic variates.

Custom — Context7 found no library equivalent for this paper-specific setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .simulation import simulate_model_paths
from .blackscholes import (
    put_price_from_mc,
    implied_vol_from_price,
    digital_probability,
    atm_implied_vol,
    atm_skew_from_digital,
)
from .coefficients import (
    compute_scenario_coefficients,
    compute_bar_v,
    compute_c_theta,
    volterra_variance,
)
from .malliavin import (
    compute_malliavin_factor_direction,
    compute_malliavin_residual_directions,
    compute_basket_malliavin,
    compute_A0_theta,
)

_phi = norm.pdf
_Phi = norm.cdf

# Standard maturity ladder: theta_m = 2^{-m} for m = 6, ..., 12
MATURITY_LADDER = np.array([2.0 ** (-m) for m in range(6, 13)])
Z_GRID = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

# Default simulation parameters
DEFAULT_M = 256          # time steps per maturity
DEFAULT_N_HALF = 5000    # antithetic path pairs per batch
DEFAULT_N_BATCHES = 50   # independent batches


def _batch_put_price(
    config: dict,
    H: float,
    theta: float,
    z_grid: np.ndarray,
    M: int,
    N_half: int,
    batch_seed: int,
) -> dict:
    """Single batch: simulate, price puts at all strikes, return prices and digital."""
    S0_basket = float(np.asarray(config["S0"]).sum())
    result = simulate_model_paths(config, H, theta, M, N_half, seed=batch_seed)
    S_terminal = result["S_basket"]  # (N,)

    put_means = []
    put_ses = []
    for z in z_grid:
        k = np.sqrt(theta) * z
        K = S0_basket * np.exp(k)
        p, se = put_price_from_mc(S_terminal, K)
        put_means.append(p)
        put_ses.append(se)

    dig, dig_se = digital_probability(S_terminal, S0_basket)

    return {
        "put_means": np.array(put_means),
        "put_ses": np.array(put_ses),
        "digital": dig,
        "digital_se": dig_se,
    }


def run_smile_experiment(
    config: dict,
    H: float,
    theta: float,
    z_grid: np.ndarray = Z_GRID,
    M: int = DEFAULT_M,
    N_half: int = DEFAULT_N_HALF,
    n_batches: int = DEFAULT_N_BATCHES,
    base_seed: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run near-the-money smile experiment for one (config, H, theta) triple.

    Estimates MC implied vols and compares to full/partial asymptotic benchmarks.

    Parameters
    ----------
    config : dict from baseline_A_config() or baseline_B_config()
    H : float
    theta : float
    z_grid : ndarray of log-moneyness z values
    M, N_half, n_batches, base_seed : simulation params
    verbose : bool

    Returns
    -------
    df : DataFrame with columns:
        z, sigma_MC, sigma_full, sigma_A_only, sigma_B_only,
        abs_error, scaled_residual, se_MC, ci_lo, ci_hi
    """
    coeffs = compute_scenario_coefficients(config, H)
    S0_basket = float(np.asarray(config["S0"]).sum())

    # Batch aggregation for each z
    batch_put_prices = [[] for _ in z_grid]
    batch_digitals = []

    for b in range(n_batches):
        if verbose and b % 10 == 0:
            print(f"    Batch {b+1}/{n_batches}")

        seed = base_seed + b
        batch = _batch_put_price(config, H, theta, z_grid, M, N_half, seed)
        for iz in range(len(z_grid)):
            batch_put_prices[iz].append(batch["put_means"][iz])
        batch_digitals.append(batch["digital"])

    # Compute mean put prices and standard errors across batches
    mean_prices = np.array([np.mean(bp) for bp in batch_put_prices])
    se_prices = np.array([np.std(bp, ddof=1) / np.sqrt(n_batches) for bp in batch_put_prices])

    # Convert to implied vols
    sigma_MC = np.full(len(z_grid), np.nan)
    sigma_se = np.full(len(z_grid), np.nan)

    for iz, z in enumerate(z_grid):
        k = np.sqrt(theta) * z
        K = S0_basket * np.exp(k)
        sigma_MC[iz] = implied_vol_from_price(mean_prices[iz], S0_basket, K, theta)
        # Delta-method: se_sigma ≈ se_price / |d(price)/d(sigma)|
        # For ATM: d(price)/d(sigma) ≈ S0 * phi(0) * sqrt(theta)
        sigma_se[iz] = se_prices[iz] / max(S0_basket * _phi(0) * np.sqrt(theta), 1e-10)

    # Asymptotic benchmarks
    bar_v = compute_bar_v(coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta)
    r_theta = theta ** H
    c_theta_full = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
    c_theta_A = compute_c_theta(coeffs["psi_A"], 0.0, H, theta)
    c_theta_B = compute_c_theta(0.0, coeffs["psi_B"], H, theta)

    sqrt_barv = np.sqrt(bar_v)
    sigma_full = sqrt_barv + c_theta_full * z_grid
    sigma_A_only = sqrt_barv + c_theta_A * z_grid
    sigma_B_only = sqrt_barv + c_theta_B * z_grid

    abs_error = sigma_MC - sigma_full
    scaled_residual = abs_error / (r_theta + np.sqrt(theta))

    rows = []
    for iz, z in enumerate(z_grid):
        ci_half = 1.96 * sigma_se[iz]
        rows.append({
            "z": z,
            "sigma_MC": sigma_MC[iz],
            "sigma_full": sigma_full[iz],
            "sigma_A_only": sigma_A_only[iz],
            "sigma_B_only": sigma_B_only[iz],
            "abs_error": abs_error[iz],
            "scaled_residual": scaled_residual[iz],
            "se_MC": sigma_se[iz],
            "ci_lo": sigma_MC[iz] - ci_half,
            "ci_hi": sigma_MC[iz] + ci_half,
        })

    return pd.DataFrame(rows)


def run_atm_skew_experiment(
    config: dict,
    H: float,
    theta: float,
    M: int = DEFAULT_M,
    N_half: int = DEFAULT_N_HALF,
    n_batches: int = DEFAULT_N_BATCHES,
    base_seed: int = 0,
    fd_h: float = 0.5,
) -> dict:
    """
    Run ATM skew experiment for one (config, H, theta) triple.

    Computes:
    - ATM IV sigma(0, theta) from ATM put
    - Digital probability Q(S_theta < S_0)
    - Digital correction Y_0^theta = Q(...) - Phi(sqrt(theta) sigma_ATM / 2)
    - Paper-faithful ATM skew via digital identity
    - Secondary FD skew estimator

    Parameters
    ----------
    config, H, theta : scenario
    M, N_half, n_batches, base_seed : simulation parameters
    fd_h : float - FD parameter (strike grid h * sqrt(theta))

    Returns
    -------
    dict with all ATM skew quantities, batch means and standard errors.
    """
    coeffs = compute_scenario_coefficients(config, H)
    S0_basket = float(np.asarray(config["S0"]).sum())

    # For FD: strikes at ±fd_h * sqrt(theta) log-moneyness
    k_fd_plus = fd_h * np.sqrt(theta)
    k_fd_minus = -fd_h * np.sqrt(theta)
    K_fd_plus = S0_basket * np.exp(k_fd_plus)
    K_fd_minus = S0_basket * np.exp(k_fd_minus)

    batch_atm_iv = []
    batch_digital = []
    batch_fd_plus = []
    batch_fd_minus = []

    for b in range(n_batches):
        seed = base_seed + b
        result = simulate_model_paths(config, H, theta, M, N_half, seed=seed)
        S_t = result["S_basket"]

        # ATM IV
        sigma_atm = atm_implied_vol(S_t, S0_basket, theta)
        batch_atm_iv.append(sigma_atm)

        # Digital
        dig, _ = digital_probability(S_t, S0_basket)
        batch_digital.append(dig)

        # FD put prices (using common random numbers within batch)
        p_plus, _ = put_price_from_mc(S_t, K_fd_plus)
        p_minus, _ = put_price_from_mc(S_t, K_fd_minus)
        sigma_plus = implied_vol_from_price(p_plus, S0_basket, K_fd_plus, theta)
        sigma_minus = implied_vol_from_price(p_minus, S0_basket, K_fd_minus, theta)
        batch_fd_plus.append(sigma_plus)
        batch_fd_minus.append(sigma_minus)

    sigma_atm_mean = float(np.nanmean(batch_atm_iv))
    sigma_atm_se = float(np.nanstd(batch_atm_iv, ddof=1) / np.sqrt(n_batches))
    digital_mean = float(np.mean(batch_digital))
    digital_se = float(np.std(batch_digital, ddof=1) / np.sqrt(n_batches))

    # Digital correction Y_0^theta
    w_atm = np.sqrt(theta) * sigma_atm_mean / 2.0
    Y0 = digital_mean - _Phi(w_atm)
    Y0_se = digital_se  # SE dominated by digital uncertainty

    # Digital-based ATM skew
    skew_digital = atm_skew_from_digital(digital_mean, sigma_atm_mean, theta)

    # FD skew
    sig_plus = float(np.nanmean(batch_fd_plus))
    sig_minus = float(np.nanmean(batch_fd_minus))
    skew_fd = (sig_plus - sig_minus) / (2.0 * k_fd_plus)
    skew_fd_se = float(np.nanstd(
        [(p - m) / (2 * k_fd_plus)
         for p, m in zip(batch_fd_plus, batch_fd_minus)
         if not (np.isnan(p) or np.isnan(m))],
        ddof=1,
    ) / np.sqrt(n_batches))

    # Theoretical ATM skew: (psi_A/(2H+3)) r(theta)/sqrt(theta) + psi_B/4
    r_theta = theta ** H
    skew_theory = (coeffs["psi_A"] / (2*H + 3)) * r_theta / np.sqrt(theta) + coeffs["psi_B"] / 4.0

    # Theoretical Y_0^theta coefficient
    c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
    Y0_theory = _phi(0.0) * c_theta

    return {
        "theta": theta,
        "H": H,
        "sigma_ATM": sigma_atm_mean,
        "sigma_ATM_se": sigma_atm_se,
        "digital": digital_mean,
        "digital_se": digital_se,
        "Y0": Y0,
        "Y0_se": Y0_se,
        "Y0_theory": Y0_theory,
        "skew_digital": skew_digital,
        "skew_fd": skew_fd,
        "skew_fd_se": skew_fd_se,
        "skew_theory": skew_theory,
        "r_theta": r_theta,
        "psi_A": coeffs["psi_A"],
        "psi_B": coeffs["psi_B"],
        "c_theta": c_theta,
    }


def run_ssr_experiment(
    config: dict,
    H: float,
    theta: float,
    M: int = DEFAULT_M,
    N_half: int = DEFAULT_N_HALF,
    n_batches: int = DEFAULT_N_BATCHES,
    base_seed: int = 0,
) -> dict:
    """
    Run SSR experiment for one (config, H, theta) triple.

    Estimates A_0^theta (Malliavin numerator) and Y_0^theta (digital denominator)
    and their ratio SSR_0(theta) = A_0^theta / Y_0^theta.

    Parameters
    ----------
    config, H, theta : scenario
    M, N_half, n_batches : simulation params
    base_seed : int

    Returns
    -------
    dict with A0, Y0, SSR, standard errors, and theoretical benchmarks.
    """
    coeffs = compute_scenario_coefficients(config, H)
    S0_basket = float(np.asarray(config["S0"]).sum())
    Pi0 = coeffs["Pi0"]
    V0 = np.asarray(config["V0"])
    rho_loadings = np.asarray(config["rho_loadings"])
    beta = np.asarray(config["beta"])
    d = config["d"]
    v0 = coeffs["v0"]

    batch_A0 = []
    batch_Y0 = []

    for b in range(n_batches):
        seed = base_seed + b
        result = simulate_model_paths(config, H, theta, M, N_half, seed=seed)
        S_t = result["S_basket"]
        V_paths = result["V_paths"]
        dB_paths = result["dB_paths"]
        Pi_terminal = result["Pi_terminal"]
        N = result["N"]
        dt = result["dt"]

        # ATM IV for Y_0^theta computation
        sigma_atm = atm_implied_vol(S_t, S0_basket, theta)
        if np.isnan(sigma_atm):
            continue

        # Digital Y_0^theta
        dig, _ = digital_probability(S_t, S0_basket)
        w_atm = np.sqrt(theta) * sigma_atm / 2.0
        Y0 = dig - _Phi(w_atm)
        batch_Y0.append(Y0)

        # Malliavin derivatives
        D1 = compute_malliavin_factor_direction(V_paths, dB_paths, beta, rho_loadings, V0, H, dt)
        D_res = compute_malliavin_residual_directions(rho_loadings, V0, d, N)
        D_basket_factor, D_basket_residual = compute_basket_malliavin(D1, D_res, Pi_terminal)

        A0_b, _ = compute_A0_theta(
            S_t, S0_basket, Pi_terminal,
            D_basket_factor, D_basket_residual,
            Pi0, V0, rho_loadings, d, v0,
        )
        batch_A0.append(A0_b)

    n_valid = len(batch_A0)
    if n_valid < 2:
        return {"error": "Too few valid batches", "theta": theta, "H": H}

    A0_mean = float(np.mean(batch_A0))
    A0_se = float(np.std(batch_A0, ddof=1) / np.sqrt(n_valid))
    Y0_mean = float(np.mean(batch_Y0))
    Y0_se = float(np.std(batch_Y0, ddof=1) / np.sqrt(n_valid))

    SSR = A0_mean / Y0_mean if abs(Y0_mean) > 1e-12 else np.nan
    SSR_target = H + 1.5

    # Theoretical terms
    r_theta = theta ** H
    psi_A = coeffs["psi_A"]
    psi_B = coeffs["psi_B"]
    A0_theory = (_phi(0.0) / 2.0) * (psi_A * r_theta + psi_B * np.sqrt(theta))
    Y0_theory = _phi(0.0) * ((psi_A / (2*H + 3)) * r_theta + (psi_B / 4.0) * np.sqrt(theta))

    return {
        "theta": theta,
        "H": H,
        "A0": A0_mean,
        "A0_se": A0_se,
        "Y0": Y0_mean,
        "Y0_se": Y0_se,
        "SSR": SSR,
        "SSR_target": SSR_target,
        "A0_theory": A0_theory,
        "Y0_theory": Y0_theory,
        "psi_A": psi_A,
        "psi_B": psi_B,
        "r_theta": r_theta,
        "n_valid_batches": n_valid,
    }


def run_density_experiment(
    config: dict,
    H: float,
    theta: float,
    x_grid: np.ndarray | None = None,
    M: int = DEFAULT_M,
    N_half: int = DEFAULT_N_HALF,
    n_batches: int = DEFAULT_N_BATCHES,
    base_seed: int = 0,
) -> dict:
    """
    Run density expansion experiment for one (config, H, theta) triple.

    Validates q_theta(x) = phi(x + sigma_{0,theta}/2) * [1 + c_theta H_3(x + sigma_{0,theta}/2)].

    Parameters
    ----------
    config, H, theta : scenario
    x_grid : ndarray - evaluation points for normalized return
    M, N_half, n_batches, base_seed : simulation params

    Returns
    -------
    dict with empirical density vs theoretical density comparison, Y_0^theta.
    """
    if x_grid is None:
        x_grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    coeffs = compute_scenario_coefficients(config, H)
    S0_basket = float(np.asarray(config["S0"]).sum())

    # Collect all terminal log-basket returns and digitals across batches
    all_log_returns = []
    batch_digitals = []
    batch_atm_ivs = []

    for b in range(n_batches):
        seed = base_seed + b
        result = simulate_model_paths(config, H, theta, M, N_half, seed=seed)
        S_t = result["S_basket"]

        log_ret = np.log(S_t / S0_basket)
        all_log_returns.append(log_ret)
        dig, _ = digital_probability(S_t, S0_basket)
        batch_digitals.append(dig)
        batch_atm_ivs.append(atm_implied_vol(S_t, S0_basket, theta))

    # Normalization: sigma_{0,theta}^2 = integral_0^theta E[V_t] dt
    # Approximated using frozen-weight proxy bar v(theta) * theta
    from .coefficients import compute_bar_v as _bar_v
    bar_v = _bar_v(coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta)
    sigma_0_theta = np.sqrt(bar_v * theta)

    # Normalized log-returns
    all_log_arr = np.concatenate(all_log_returns)
    L_theta = all_log_arr / sigma_0_theta  # normalized basket log-return

    # Theoretical density
    c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
    # q_theta(x) = phi(x + sigma_0/2) * [1 + c_theta * H_3(x + sigma_0/2)]
    # H_3(u) = u^3 - 3u

    def q_theta(x_vals: np.ndarray) -> np.ndarray:
        u = x_vals + sigma_0_theta / 2.0
        H3 = u**3 - 3.0 * u
        return _phi(u) * (1.0 + c_theta * H3)

    # Empirical density via histogram (transparent primary estimator)
    bin_width = 0.5
    bins = np.arange(-3.5, 3.5 + bin_width, bin_width)
    hist_counts, bin_edges = np.histogram(L_theta, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Evaluate at x_grid
    q_theory_at_x = q_theta(x_grid)

    # Interpolate empirical density at x_grid
    q_empirical_at_x = np.interp(x_grid, bin_centers, hist_counts, left=np.nan, right=np.nan)

    # Digital summary
    digital_mean = float(np.mean(batch_digitals))
    digital_se = float(np.std(batch_digitals, ddof=1) / np.sqrt(n_batches))
    sigma_atm_mean = float(np.nanmean(batch_atm_ivs))

    w_atm = np.sqrt(theta) * sigma_atm_mean / 2.0
    Y0 = digital_mean - _Phi(w_atm)
    Y0_theory = _phi(0.0) * c_theta

    return {
        "theta": theta,
        "H": H,
        "sigma_0_theta": sigma_0_theta,
        "c_theta": c_theta,
        "x_grid": x_grid,
        "q_theory": q_theory_at_x,
        "q_empirical": q_empirical_at_x,
        "bin_centers": bin_centers,
        "hist_density": hist_counts,
        "density_error": q_empirical_at_x - q_theory_at_x,
        "digital_mean": digital_mean,
        "digital_se": digital_se,
        "sigma_atm": sigma_atm_mean,
        "Y0": Y0,
        "Y0_theory": Y0_theory,
        "psi_A": coeffs["psi_A"],
        "psi_B": coeffs["psi_B"],
    }
