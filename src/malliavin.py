"""
Malliavin derivative computation for the SSR (Skew Stickiness Ratio) experiment.

Implements the one-factor simplification from the paper:
  D_0^1 log S_theta^j = rho_1^j sqrt(V_0^j) + 0.5 int_0^theta sqrt(V_s^j) beta_j K(s) dB_s^j
                       - 0.5 int_0^theta V_s^j beta_j K(s) ds
  D_0^l log S_theta^j = rho_l^j sqrt(V_0^j)  for l = d+1, ..., d+n (residual directions)
  D_0^l log S_theta = sum_j Pi_theta^j D_0^l log S_theta^j

Custom — Context7 found no library equivalent (paper-specific Malliavin formula).
"""

from __future__ import annotations

import numpy as np


def compute_malliavin_factor_direction(
    V_paths: np.ndarray,
    dB_paths: np.ndarray,
    beta: np.ndarray,
    rho_loadings: np.ndarray,
    V0: np.ndarray,
    H: float,
    dt: float,
) -> np.ndarray:
    """
    Compute D_0^1 log S_theta^j for each constituent j and path.

    D_0^1 log S_theta^j = rho_1^j sqrt(V_0^j)
        + 0.5 * int_0^theta sqrt(V_s^j) * beta_j * K(s) * dB_s^j
        - 0.5 * int_0^theta V_s^j * beta_j * K(s) * ds

    where K(s) = s^{H - 0.5} is the pure-power Volterra kernel.

    Discretized using left-point sums:
      stoch_int ≈ sum_m sqrt(V_{t_m}^j) * beta_j * K(t_m) * dB_{t_m}^j
      lebesgue_int ≈ sum_m V_{t_m}^j * beta_j * K(t_m) * dt

    Parameters
    ----------
    V_paths : ndarray of shape (n, M+1, N)
        Variance paths. Index [i, m, path].
    dB_paths : ndarray of shape (n, M, N)
        Return Brownian increments dB_t^i.
    beta : ndarray of shape (n,)
        Vol-of-vol sensitivities.
    rho_loadings : ndarray of shape (n, d+n)
        Loading matrix.
    V0 : ndarray of shape (n,)
        Initial variances.
    H : float
        Hurst parameter.
    dt : float
        Time step.

    Returns
    -------
    D1 : ndarray of shape (n, N)
        D_0^1 log S_theta^j for each constituent j.
    """
    n, M_plus1, N = V_paths.shape
    M = M_plus1 - 1

    # Kernel K(t_m) = t_m^{H - 0.5} evaluated at left-points t_m = m * dt, m = 1, ..., M
    t_grid = np.arange(1, M + 1, dtype=float) * dt  # (M,)
    K_vals = t_grid ** (H - 0.5)  # (M,)

    D1 = np.zeros((n, N))

    for j in range(n):
        # Constant term
        const_term = rho_loadings[j, 0] * np.sqrt(V0[j])  # scalar

        # V_{t_m}^j : shape (M,) for left-point at m=0..M-1, or (M+1,) using V[:M]
        V_j = V_paths[j, :M, :]  # (M, N) - values at t_0, ..., t_{M-1}
        dB_j = dB_paths[j, :, :]  # (M, N)

        # Stochastic integral: sum_m sqrt(V_{t_m}^j) * beta_j * K(t_m) * dB_{t_m}^j
        # K_vals shape: (M,), need (M, 1) to broadcast over N
        stoch_integrand = (
            np.sqrt(V_j)                # (M, N)
            * beta[j]
            * K_vals[:, None]           # (M, 1)
            * dB_j                      # (M, N)
        )
        stoch_int = stoch_integrand.sum(axis=0)  # (N,)

        # Lebesgue integral: sum_m V_{t_m}^j * beta_j * K(t_m) * dt
        lebesgue_integrand = (
            V_j                         # (M, N)
            * beta[j]
            * K_vals[:, None]           # (M, 1)
            * dt
        )
        lebesgue_int = lebesgue_integrand.sum(axis=0)  # (N,)

        D1[j] = const_term + 0.5 * stoch_int - 0.5 * lebesgue_int

    return D1


def compute_malliavin_residual_directions(
    rho_loadings: np.ndarray,
    V0: np.ndarray,
    d: int,
    N: int,
) -> np.ndarray:
    """
    Compute D_0^l log S_theta^j = rho_l^j sqrt(V_0^j) for residual directions l = d+1..d+n.

    These are path-independent constants.

    Parameters
    ----------
    rho_loadings : ndarray of shape (n, d+n)
    V0 : ndarray of shape (n,)
    d : int
    N : int
        Number of paths (returned as broadcast-compatible array).

    Returns
    -------
    D_residual : ndarray of shape (n, n_residual, N)
        D_residual[j, l_idx, :] = rho_{d+l_idx+1}^j * sqrt(V_0^j)
    """
    n = len(V0)
    n_residual = rho_loadings.shape[1] - d  # should equal n
    sqrt_V0 = np.sqrt(V0)

    # D[j, l_idx] = rho_loadings[j, d + l_idx] * sqrt_V0[j]
    D_const = rho_loadings[:, d:] * sqrt_V0[:, None]  # (n, n_residual)

    # Broadcast to (n, n_residual, N)
    D_residual = np.broadcast_to(D_const[:, :, None], (n, n_residual, N)).copy()
    return D_residual


def compute_basket_malliavin(
    D1_constituent: np.ndarray,
    D_residual: np.ndarray,
    Pi_terminal: np.ndarray,
) -> np.ndarray:
    """
    Compute basket Malliavin derivatives:
    D_0^l log S_theta = sum_j Pi_theta^j D_0^l log S_theta^j.

    Parameters
    ----------
    D1_constituent : ndarray of shape (n, N)
        Factor direction Malliavin derivatives.
    D_residual : ndarray of shape (n, n_residual, N)
        Residual direction derivatives.
    Pi_terminal : ndarray of shape (n, N)
        Terminal basket weights.

    Returns
    -------
    D_basket_factor : ndarray of shape (N,)
        sum_j Pi_theta^j D_0^1 log S_theta^j
    D_basket_residual : ndarray of shape (n_residual, N)
        sum_j Pi_theta^j D_0^l log S_theta^j for each residual l.
    """
    # Factor direction (l=1)
    D_basket_factor = (Pi_terminal * D1_constituent).sum(axis=0)  # (N,)

    # Residual directions
    D_basket_residual = np.einsum("jN,jlN->lN", Pi_terminal, D_residual)  # (n_res, N)

    return D_basket_factor, D_basket_residual


def compute_A0_theta(
    S_basket_terminal: np.ndarray,
    S0_basket: float,
    Pi_terminal: np.ndarray,
    D_basket_factor: np.ndarray,
    D_basket_residual: np.ndarray,
    Pi0: np.ndarray,
    V0: np.ndarray,
    rho_loadings: np.ndarray,
    d: int,
    v0: float,
) -> tuple[float, float]:
    """
    Compute A_0^theta = E[1_{S_theta < S_0} (S_theta/S_0)
        (1 - (1/V_0) sum_{l,i} Pi_0^i sqrt(V_0^i) rho_l^i D_0^l log S_theta)]

    where V_0 = v(0) is the basket variance at time 0.

    Parameters
    ----------
    S_basket_terminal : (N,)
    S0_basket : float
    Pi_terminal : (n, N) - terminal weights Pi_theta^i
    D_basket_factor : (N,) - basket Malliavin in factor direction l=1
    D_basket_residual : (n_residual, N) - basket Malliavin in residual directions
    Pi0 : (n,) - initial weights
    V0 : (n,) - initial variances
    rho_loadings : (n, d+n)
    d : int
    v0 : float - v(0) = basket variance at t=0

    Returns
    -------
    A0_mean : float
    A0_se : float
    """
    N = len(S_basket_terminal)
    n = len(Pi0)
    sqrt_V0 = np.sqrt(V0)

    # Malliavin weight: sum_{l,i} Pi_0^i sqrt(V_0^i) rho_l^i D_0^l log S_theta
    # l=1 (factor direction):
    # sum_i Pi_0^i sqrt(V_0^i) rho_1^i * D_basket_factor
    weight_factor = float(np.dot(Pi0 * sqrt_V0, rho_loadings[:, 0]))  # scalar
    malliavin_term_factor = weight_factor * D_basket_factor  # (N,)

    # l = d+1..d+n (residual directions):
    # sum_i Pi_0^i sqrt(V_0^i) rho_{d+l_idx+1}^i * D_basket_residual[l_idx]
    # weights_res shape: (n_residual,)
    weights_res = Pi0 * sqrt_V0 @ rho_loadings[:, d:]  # (n_residual,)
    malliavin_term_residual = (weights_res[:, None] * D_basket_residual).sum(axis=0)  # (N,)

    malliavin_total = malliavin_term_factor + malliavin_term_residual  # (N,)

    # Pathwise integrand
    indicator = (S_basket_terminal < S0_basket).astype(float)
    ratio = S_basket_terminal / S0_basket
    integrand = indicator * ratio * (1.0 - malliavin_total / v0)  # (N,)

    A0_mean = float(integrand.mean())
    A0_se = float(integrand.std(ddof=1) / np.sqrt(N))
    return A0_mean, A0_se
