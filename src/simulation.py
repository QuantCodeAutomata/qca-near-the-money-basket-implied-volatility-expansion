"""
Monte Carlo simulation for Gaussian Volterra-factor basket models.

Implements:
  - Volterra factor Z_t = integral_0^t (t-s)^{H-1/2} dW_s^1
  - Variance paths V_t^i = V_0^i exp(beta_i Z_t - 0.5 beta_i^2 Var(Z_t))
  - Asset paths via log-Euler: log S_{t+dt}^i = log S_t^i - 0.5 V_t^i dt + sqrt(V_t^i) dB_t^i
  - Basket S_t = sum_i S_t^i and endogenous weights Pi_t^i = S_t^i / S_t

Custom — Context7 found no library equivalent (paper-specific Gaussian Volterra model).
"""

from __future__ import annotations

import numpy as np
from .coefficients import volterra_variance


def build_volterra_kernel_matrix(M: int, dt: float, H: float) -> np.ndarray:
    """
    Build lower-triangular kernel matrix for Volterra convolution using a
    hybrid midpoint + exact-isometry-first-lag scheme.

    For the last (most singular) lag ``diff == 1``, the exact isometric weight
    is used so that Var(K[m,m-1] * dW) = dt^{2H}/(2H) exactly.
    For farther lags ``diff >= 2``, the midpoint rule is used:
    K[m,j] = ((m-j-0.5)*dt)^{H-0.5}.

    This eliminates the O(1) bias that the plain left-point rule has for H < 0.5.

    Parameters
    ----------
    M : int
        Number of time steps.
    dt : float
        Time step size.
    H : float
        Hurst exponent in (0, 1/2].

    Returns
    -------
    K_mat : ndarray of shape (M, M)
        Row m-1, column j gives the weight applied to dW1[j] to form Z[m].
    """
    m_idx = np.arange(1, M + 1, dtype=float)[:, None]  # (M, 1)
    j_idx = np.arange(M, dtype=float)[None, :]           # (1, M)
    diff = m_idx - j_idx                                  # integer lags (M, M)

    # Midpoint weights for lags >= 2: use (diff - 0.5) * dt as the evaluation point.
    # Clip lag values to >= 1.5 before power to avoid NaN from np.where eager evaluation.
    safe_mid = np.clip(diff - 0.5, 1.5, None)
    K_mid = np.where(diff >= 2, (safe_mid * dt) ** (H - 0.5), 0.0)

    # Exact-isometry weight for lag == 1:
    # Var of integral_0^dt u^{H-1/2} dW_u = dt^{2H}/(2H)
    # => weight applied to dW (std sqrt(dt)) must be dt^{H-0.5}/sqrt(2H)
    K_exact_lag1 = np.where(diff == 1, dt ** (H - 0.5) / np.sqrt(2.0 * H), 0.0)

    K_mat = K_mid + K_exact_lag1
    return K_mat


def simulate_volterra_factor(
    dW1: np.ndarray,
    H: float,
    dt: float,
    K_mat: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simulate the Volterra factor Z_t = int_0^t (t-s)^{H-1/2} dW_s^1.

    Uses left-point Riemann sum discretization:
    Z_{t_m} = sum_{j=0}^{m-1} (t_m - t_j)^{H-1/2} * dW_j^1.

    Parameters
    ----------
    dW1 : ndarray of shape (M, N)
        Brownian increments for factor driver (mean 0, std sqrt(dt)).
    H : float
        Hurst parameter in (0, 1/2].
    dt : float
        Time step.
    K_mat : ndarray of shape (M, M), optional
        Pre-computed kernel matrix. Computed if None.

    Returns
    -------
    Z : ndarray of shape (M+1, N) with Z[0] = 0.
    """
    M, N = dW1.shape
    if K_mat is None:
        K_mat = build_volterra_kernel_matrix(M, dt, H)

    # Z[1:] = K_mat @ dW1 with shape (M, N)
    Z_inner = K_mat @ dW1
    Z = np.vstack([np.zeros((1, N)), Z_inner])
    return Z


def simulate_variance_paths(
    Z: np.ndarray,
    V0: np.ndarray,
    beta: np.ndarray,
    H: float,
    dt: float,
) -> np.ndarray:
    """
    Simulate variance paths V_t^i = V_0^i exp(beta_i Z_t - 0.5 beta_i^2 Var(Z_t)).

    E[V_t^i] = V_0^i exactly due to the martingale correction.

    Parameters
    ----------
    Z : ndarray of shape (M+1, N)
        Volterra factor paths.
    V0 : ndarray of shape (n,)
        Initial variances.
    beta : ndarray of shape (n,)
        Vol-of-vol sensitivities.
    H : float
        Hurst exponent.
    dt : float
        Time step size.

    Returns
    -------
    V : ndarray of shape (n, M+1, N)
    """
    M_plus1 = Z.shape[0]
    N = Z.shape[1]
    n = len(V0)

    # Time grid for variance correction
    t_grid = np.arange(M_plus1, dtype=float) * dt  # (M+1,)
    var_Zt = volterra_variance(np.maximum(t_grid, 1e-30), H)  # (M+1,)
    var_Zt[0] = 0.0  # Z_0 = 0 surely

    # V[i, m, path] = V0[i] * exp(beta[i] * Z[m, path] - 0.5 * beta[i]^2 * var_Zt[m])
    # Shape broadcasts: (n,1,1) * (1,M+1,N)
    beta_col = beta[:, None, None]         # (n, 1, 1)
    V0_col = V0[:, None, None]             # (n, 1, 1)
    var_col = var_Zt[None, :, None]        # (1, M+1, 1)
    Z_expand = Z[None, :, :]              # (1, M+1, N)

    V = V0_col * np.exp(beta_col * Z_expand - 0.5 * beta_col**2 * var_col)
    return V


def simulate_asset_paths(
    V: np.ndarray,
    dB: np.ndarray,
    S0: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Simulate log-Euler asset paths:
    log S_{t+dt}^i = log S_t^i - 0.5 * V_t^i * dt + sqrt(V_t^i) * dB_t^i.

    Parameters
    ----------
    V : ndarray of shape (n, M+1, N)
        Variance paths. Only V[:, :M, :] used (variance at start of each interval).
    dB : ndarray of shape (n, M, N)
        Brownian increments for return drivers (mean 0, std sqrt(dt)).
    S0 : ndarray of shape (n,)
        Initial asset prices.
    dt : float

    Returns
    -------
    S : ndarray of shape (n, M+1, N)
        Asset price paths with S[:, 0, :] = S0.
    """
    n, M_plus1, N = V.shape
    M = M_plus1 - 1

    log_S = np.zeros((n, M + 1, N))
    log_S[:, 0, :] = np.log(S0)[:, None]

    for m in range(M):
        v_t = V[:, m, :]        # (n, N) - variance at step m
        db_t = dB[:, m, :]      # (n, N)
        log_S[:, m + 1, :] = (
            log_S[:, m, :]
            - 0.5 * v_t * dt
            + np.sqrt(v_t) * db_t
        )

    S = np.exp(log_S)
    return S


def simulate_model_paths(
    config: dict,
    H: float,
    theta: float,
    M: int,
    N_half: int,
    seed: int | None = None,
) -> dict:
    """
    Full simulation of the Gaussian Volterra-factor basket model up to maturity theta.

    Uses antithetic variates: total N = 2 * N_half paths.

    Parameters
    ----------
    config : dict
        Baseline configuration from baseline_A_config() or baseline_B_config().
    H : float
        Hurst parameter.
    theta : float
        Maturity.
    M : int
        Number of time steps.
    N_half : int
        Number of antithetic path pairs (total paths = 2 * N_half).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    results : dict with keys:
        - S_terminal : ndarray (n, N) - terminal constituent prices
        - S_basket : ndarray (N,) - terminal basket prices
        - Pi_terminal : ndarray (n, N) - terminal weights
        - S_paths : ndarray (n, M+1, N) - full price paths
        - V_paths : ndarray (n, M+1, N) - full variance paths
        - Z_paths : ndarray (M+1, N) - Volterra factor paths
        - dB_paths : ndarray (n, M, N) - return Brownian increments
        - dW_paths : ndarray (d+n, M, N) - factor Brownian increments
        - dt : float
        - M : int
        - N : int - total path count
        - theta : float
        - config : dict
        - H : float
    """
    rng = np.random.default_rng(seed)

    n = config["n"]
    d = config["d"]
    S0 = np.asarray(config["S0"], dtype=float)
    V0 = np.asarray(config["V0"], dtype=float)
    beta = np.asarray(config["beta"], dtype=float)
    rho_loadings = np.asarray(config["rho_loadings"], dtype=float)  # (n, d+n)

    dt = theta / M
    N = 2 * N_half  # total including antithetic

    # Build Volterra kernel matrix once
    K_mat = build_volterra_kernel_matrix(M, dt, H)

    # Generate independent standard normal increments for W^1, ..., W^{d+n}
    # Shape: (d+n, M, N_half)
    xi_half = rng.standard_normal((d + n, M, N_half))
    # Antithetic: mirror all increments
    xi_anti = -xi_half
    xi = np.concatenate([xi_half, xi_anti], axis=2)  # (d+n, M, N)

    # Brownian increments dW = sqrt(dt) * xi, shape (d+n, M, N)
    dW = np.sqrt(dt) * xi

    # Factor Brownian increments (first d factors drive the Volterra kernel)
    # For one-factor model: dW1 = dW[0, :, :]
    dW1 = dW[0]  # (M, N)

    # Simulate Volterra factor
    Z = simulate_volterra_factor(dW1, H, dt, K_mat)  # (M+1, N)

    # Simulate variance paths
    V = simulate_variance_paths(Z, V0, beta, H, dt)  # (n, M+1, N)

    # Construct return driver increments: dB^i = R @ dW
    # rho_loadings shape: (n, d+n), dW shape: (d+n, M, N)
    # dB[i, m, :] = sum_l rho_loadings[i, l] * dW[l, m, :]
    # = einsum("il,lmN->imN")
    dB = np.einsum("il,lmN->imN", rho_loadings, dW)  # (n, M, N)

    # Simulate asset paths
    S = simulate_asset_paths(V, dB, S0, dt)  # (n, M+1, N)

    # Basket and terminal weights
    S_terminal = S[:, -1, :]          # (n, N)
    S_basket = S_terminal.sum(axis=0) # (N,)
    Pi_terminal = S_terminal / S_basket[None, :]  # (n, N)

    return {
        "S_terminal": S_terminal,
        "S_basket": S_basket,
        "Pi_terminal": Pi_terminal,
        "S_paths": S,
        "V_paths": V,
        "Z_paths": Z,
        "dB_paths": dB,
        "dW_paths": dW,
        "dt": dt,
        "M": M,
        "N": N,
        "theta": theta,
        "config": config,
        "H": H,
    }


def simulate_model_paths_batch(
    config: dict,
    H: float,
    theta: float,
    M: int,
    N_half: int,
    n_batches: int,
    base_seed: int = 42,
    verbose: bool = False,
) -> list[dict]:
    """
    Run multiple independent simulation batches.

    Parameters
    ----------
    config, H, theta, M, N_half : as in simulate_model_paths
    n_batches : int
        Number of independent batches.
    base_seed : int
        Base seed; batch b uses seed base_seed + b.
    verbose : bool

    Returns
    -------
    batches : list of result dicts from simulate_model_paths.
    """
    batches = []
    for b in range(n_batches):
        if verbose and b % 10 == 0:
            print(f"  Batch {b+1}/{n_batches}")
        result = simulate_model_paths(config, H, theta, M, N_half, seed=base_seed + b)
        batches.append(result)
    return batches


def check_martingale(batches: list[dict], atol: float = 0.02) -> dict:
    """
    Verify E[S_theta^i / S_0^i] ≈ 1 (martingale condition).

    Parameters
    ----------
    batches : list of simulation result dicts.
    atol : float
        Tolerance for martingale check.

    Returns
    -------
    results : dict with mean ratios and pass/fail flags.
    """
    config = batches[0]["config"]
    S0 = np.asarray(config["S0"])
    n = config["n"]

    ratios_all = []
    for b in batches:
        ratios = b["S_terminal"] / S0[:, None]  # (n, N)
        ratios_all.append(ratios.mean(axis=1))  # (n,)

    mean_ratios = np.mean(ratios_all, axis=0)  # (n,)
    passed = np.allclose(mean_ratios, 1.0, atol=atol)

    return {"mean_ratios": mean_ratios, "passed": passed, "atol": atol}
