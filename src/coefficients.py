"""
Theoretical asymptotic coefficient computation for Gaussian Volterra-factor basket models.

Implements the formulas from the paper:
  - psi^{ki} = 2 gamma^{ki} / (2H+1)
  - psi_A and psi_B basket skew decomposition
  - bar v(theta) frozen-weight variance proxy
  - r(theta) = theta^H short-maturity scaling

Custom — Context7 found no library equivalent (paper-specific asymptotics).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigh
from scipy import integrate


def compute_return_correlations(rho_loadings: np.ndarray) -> np.ndarray:
    """
    Compute return correlation matrix rho^{ij} = sum_l rho_l^i rho_l^j.

    Parameters
    ----------
    rho_loadings : ndarray of shape (n, d+n)
        Loading matrix R where R[i, l] = rho_l^{i+1}.

    Returns
    -------
    rho : ndarray of shape (n, n)
        Return correlation matrix rho^{ij}.
    """
    return rho_loadings @ rho_loadings.T


def compute_residual_covariance(rho_loadings: np.ndarray, d: int) -> np.ndarray:
    """
    Compute residual covariance C^perp_{ij} = sum_{l=d+1}^{d+n} rho_l^i rho_l^j.

    Parameters
    ----------
    rho_loadings : ndarray of shape (n, d+n)
    d : int
        Number of common (non-residual) factors.

    Returns
    -------
    C_perp : ndarray of shape (n, n)
    """
    residual = rho_loadings[:, d:]
    return residual @ residual.T


def check_positive_definite(matrix: np.ndarray) -> tuple[bool, np.ndarray]:
    """
    Check positive definiteness via eigenvalues.

    Parameters
    ----------
    matrix : ndarray of shape (n, n)

    Returns
    -------
    is_pd : bool
    eigenvalues : ndarray
    """
    eigenvalues = eigh(matrix)[0]
    return bool(np.all(eigenvalues > 0)), eigenvalues


def compute_gamma(
    beta: np.ndarray,
    rho_loadings: np.ndarray,
    kappa: float = 1.0,
    q: int = 1,
    d: int = 1,
) -> np.ndarray:
    """
    Compute gamma^{ki} = sum_{a=1}^q sum_{l=1}^d beta_a^i * rho_l^k * kappa_{al}.

    In the one-factor pure-power case (q=1, d=1, kappa=1):
      gamma^{ki} = beta_i * rho_1^k

    Parameters
    ----------
    beta : ndarray of shape (n,)
        beta_i = partial_{z} log f^i(0, 0) for each constituent i.
    rho_loadings : ndarray of shape (n, d+n)
        Loading matrix. rho_loadings[k, l] = rho_l^{k+1}.
    kappa : float
        Kernel-factor coupling constant (=1 for pure-power kernels).
    q : int
        Number of Volterra factors.
    d : int
        Number of common Brownian drivers.

    Returns
    -------
    gamma : ndarray of shape (n, n)
        gamma[k, i] = gamma^{ki}.
    """
    assert q == 1 and d == 1, "Only one-factor pure-power case implemented."
    # gamma^{ki} = beta_i * rho_1^k * kappa
    # beta shape: (n,), rho_1 shape: (n,)
    rho_1 = rho_loadings[:, 0]  # first common factor loading for each asset
    gamma = np.outer(rho_1, beta) * kappa  # gamma[k, i] = rho_1^k * beta_i
    return gamma


def compute_psi_ki(gamma: np.ndarray, H: float) -> np.ndarray:
    """
    Compute psi^{ki} = 2 * gamma^{ki} / (2H + 1).

    Parameters
    ----------
    gamma : ndarray of shape (n, n)
    H : float
        Hurst parameter in (0, 1/2].

    Returns
    -------
    psi_ki : ndarray of shape (n, n)
    """
    return 2.0 * gamma / (2.0 * H + 1.0)


def volterra_variance(t: float | np.ndarray, H: float) -> float | np.ndarray:
    """
    Compute Var(Z_t) = t^{2H} / (2H) for power-law Volterra kernel K(u) = u^{H-1/2}.

    Holds for all H in (0, 1/2] since t^{2*0.5}/(2*0.5) = t.

    Parameters
    ----------
    t : float or ndarray
    H : float

    Returns
    -------
    var_Zt : same shape as t
    """
    return t ** (2.0 * H) / (2.0 * H)


def compute_E_sqrt_ViVj(
    V0: np.ndarray,
    beta: np.ndarray,
    H: float,
    t: float | np.ndarray,
) -> np.ndarray:
    """
    Compute E[sqrt(V_t^i * V_t^j)] = sqrt(V_0^i * V_0^j) * exp(-((beta_i - beta_j)^2 / 8) * Var(Z_t)).

    Parameters
    ----------
    V0 : ndarray of shape (n,)
    beta : ndarray of shape (n,)
    H : float
    t : float or ndarray

    Returns
    -------
    E_matrix : ndarray of shape (n, n) or (n, n, len(t))
    """
    var_Zt = volterra_variance(t, H)  # scalar or array
    sqrt_V0 = np.sqrt(V0)
    outer_sqrt = np.outer(sqrt_V0, sqrt_V0)  # (n, n)

    # (beta_i - beta_j)^2 matrix
    beta_diff_sq = (beta[:, None] - beta[None, :]) ** 2  # (n, n)

    if np.isscalar(t):
        exp_factor = np.exp(-beta_diff_sq / 8.0 * var_Zt)
        return outer_sqrt * exp_factor
    else:
        # t is array of shape (T,)
        # result shape: (n, n, T)
        exp_factor = np.exp(-beta_diff_sq[:, :, None] / 8.0 * var_Zt[None, None, :])
        return outer_sqrt[:, :, None] * exp_factor


def compute_vt(
    Pi0: np.ndarray,
    V0: np.ndarray,
    beta: np.ndarray,
    rho: np.ndarray,
    H: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute deterministic frozen-weight basket variance:
    v(t) = sum_{i,j} Pi_0^i Pi_0^j E[sqrt(V_t^i V_t^j)] rho^{ij}.

    Parameters
    ----------
    Pi0 : ndarray of shape (n,)
    V0 : ndarray of shape (n,)
    beta : ndarray of shape (n,)
    rho : ndarray of shape (n, n)
    H : float
    t : float or ndarray

    Returns
    -------
    v_t : scalar or ndarray of shape (T,)
    """
    E_mat = compute_E_sqrt_ViVj(V0, beta, H, t)  # (n, n) or (n, n, T)
    Pi_outer = np.outer(Pi0, Pi0)  # (n, n)

    if np.isscalar(t):
        return float(np.sum(Pi_outer * rho * E_mat))
    else:
        return np.einsum("ij,ij,ijt->t", Pi_outer, rho, E_mat)


def compute_bar_v(
    Pi0: np.ndarray,
    V0: np.ndarray,
    beta: np.ndarray,
    rho: np.ndarray,
    H: float,
    theta: float,
    n_quad: int = 500,
) -> float:
    """
    Compute bar v(theta) = theta^{-1} * integral_0^theta v(t) dt by quadrature.

    Parameters
    ----------
    Pi0, V0, beta, rho : as above
    H : float
    theta : float
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    bar_v : float
    """
    # Use scipy quad on a fine grid
    t_grid = np.linspace(1e-12, theta, n_quad)
    v_grid = compute_vt(Pi0, V0, beta, rho, H, t_grid)
    bar_v = float(np.trapz(v_grid, t_grid)) / theta
    return bar_v


def compute_d_i(
    Pi0: np.ndarray,
    V0: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """
    Compute d_i = sum_j Pi_0^j sqrt(V_0^i V_0^j) rho^{ij}.

    Parameters
    ----------
    Pi0 : ndarray of shape (n,)
    V0 : ndarray of shape (n,)
    rho : ndarray of shape (n, n) - return correlation matrix

    Returns
    -------
    d : ndarray of shape (n,)
    """
    sqrt_V0 = np.sqrt(V0)
    # d[i] = sum_j Pi_0^j * sqrt(V0[i]*V0[j]) * rho[i,j]
    # = sqrt(V0[i]) * sum_j Pi_0^j * sqrt(V0[j]) * rho[i,j]
    return sqrt_V0 * (rho @ (Pi0 * sqrt_V0))


def compute_v0_scalar(
    Pi0: np.ndarray,
    V0: np.ndarray,
    rho: np.ndarray,
) -> float:
    """
    Compute v(0) = sum_{i,j} Pi_0^i Pi_0^j sqrt(V_0^i V_0^j) rho^{ij}.

    Parameters
    ----------
    Pi0 : ndarray of shape (n,)
    V0 : ndarray of shape (n,)
    rho : ndarray of shape (n, n)

    Returns
    -------
    v0 : float
    """
    d = compute_d_i(Pi0, V0, rho)
    return float(np.dot(Pi0, d))


def compute_psi_A(
    Pi0: np.ndarray,
    V0: np.ndarray,
    rho: np.ndarray,
    psi_ki: np.ndarray,
) -> float:
    """
    Compute psi_A = v(0)^{-3/2} * sum_{i,k} Pi_0^i Pi_0^k d_i sqrt(V_0^k) psi^{ki}.

    Parameters
    ----------
    Pi0 : ndarray of shape (n,)
    V0 : ndarray of shape (n,)
    rho : ndarray of shape (n, n)
    psi_ki : ndarray of shape (n, n), psi_ki[k, i]

    Returns
    -------
    psi_A : float
    """
    v0 = compute_v0_scalar(Pi0, V0, rho)
    d = compute_d_i(Pi0, V0, rho)
    sqrt_V0 = np.sqrt(V0)

    # sum_{i,k} Pi_0^i Pi_0^k d_i sqrt(V_0^k) psi^{ki}
    # = sum_i Pi_0^i d_i * sum_k Pi_0^k sqrt(V_0^k) psi^{ki}
    # psi_ki[k, i] => for each i, sum over k: sum_k Pi_0^k sqrt_V0^k psi_ki[k,i]
    inner = Pi0 * sqrt_V0  # (n,)
    sum_k = inner @ psi_ki  # (n,), sum_k Pi_0^k sqrt_V0^k psi^{ki} for each i

    numerator = float(np.dot(Pi0 * d, sum_k))
    return numerator / (v0 ** 1.5)


def compute_psi_B(
    Pi0: np.ndarray,
    V0: np.ndarray,
    rho: np.ndarray,
) -> float:
    """
    Compute psi_B = [2/v(0)^{3/2}] * (sum_i Pi_0^i d_i^2 - v(0)^2).

    Parameters
    ----------
    Pi0 : ndarray of shape (n,)
    V0 : ndarray of shape (n,)
    rho : ndarray of shape (n, n)

    Returns
    -------
    psi_B : float
    """
    v0 = compute_v0_scalar(Pi0, V0, rho)
    d = compute_d_i(Pi0, V0, rho)

    sum_pid2 = float(np.dot(Pi0, d ** 2))
    return 2.0 * (sum_pid2 - v0 ** 2) / (v0 ** 1.5)


def compute_c_theta(
    psi_A: float,
    psi_B: float,
    H: float,
    theta: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the smile slope coefficient:
    c_theta = (psi_A / (2H+3)) * r(theta) + (psi_B / 4) * sqrt(theta).

    where r(theta) = theta^H.

    Parameters
    ----------
    psi_A, psi_B : float
    H : float
    theta : float or ndarray

    Returns
    -------
    c_theta : same shape as theta
    """
    r_theta = theta ** H
    return (psi_A / (2.0 * H + 3.0)) * r_theta + (psi_B / 4.0) * np.sqrt(theta)


def compute_scenario_coefficients(config: dict, H: float) -> dict:
    """
    Compute all theoretical coefficients for a given scenario and H.

    Parameters
    ----------
    config : dict with keys:
        - n : int, number of constituents
        - d : int, number of common factors
        - q : int, number of Volterra factors (=1)
        - S0 : ndarray (n,), initial prices
        - V0 : ndarray (n,), initial variances
        - rho_loadings : ndarray (n, d+n), Brownian loadings
        - beta : ndarray (n,), vol-of-vol sensitivities
        - kappa : float (=1 for pure-power kernels)
    H : float
        Hurst parameter.

    Returns
    -------
    coeffs : dict with all computed theoretical quantities.
    """
    n = config["n"]
    d = config["d"]
    S0 = np.asarray(config["S0"], dtype=float)
    V0 = np.asarray(config["V0"], dtype=float)
    rho_loadings = np.asarray(config["rho_loadings"], dtype=float)
    beta = np.asarray(config["beta"], dtype=float)
    kappa = float(config.get("kappa", 1.0))

    Pi0 = S0 / S0.sum()

    # Return correlations
    rho = compute_return_correlations(rho_loadings)

    # Residual covariance and positive definiteness check
    C_perp = compute_residual_covariance(rho_loadings, d)
    is_pd, eigvals = check_positive_definite(C_perp)

    # Gamma and psi tensors
    gamma = compute_gamma(beta, rho_loadings, kappa=kappa, q=1, d=d)
    psi_ki = compute_psi_ki(gamma, H)

    # Basket variance components
    d_i = compute_d_i(Pi0, V0, rho)
    v0 = compute_v0_scalar(Pi0, V0, rho)

    # Asymptotic coefficients
    psi_A = compute_psi_A(Pi0, V0, rho, psi_ki)
    psi_B = compute_psi_B(Pi0, V0, rho)

    # Nondegeneracy checks
    nondegen_A = abs(psi_A) > 1e-10  # psi_A != 0 required for H < 1/2
    nondegen_AB = abs(psi_A + psi_B) > 1e-10  # psi_A + psi_B != 0 for H=1/2

    return {
        "n": n,
        "d": d,
        "H": H,
        "S0": S0,
        "V0": V0,
        "Pi0": Pi0,
        "rho_loadings": rho_loadings,
        "beta": beta,
        "kappa": kappa,
        "rho": rho,
        "C_perp": C_perp,
        "C_perp_is_pd": is_pd,
        "C_perp_eigenvalues": eigvals,
        "gamma": gamma,
        "psi_ki": psi_ki,
        "d_i": d_i,
        "v0": v0,
        "psi_A": psi_A,
        "psi_B": psi_B,
        "nondegen_A": nondegen_A,
        "nondegen_AB": nondegen_AB,
    }


# ---------------------------------------------------------------------------
# Pre-defined baseline configurations
# ---------------------------------------------------------------------------

def baseline_A_config() -> dict:
    """
    Baseline A: n=2, d=1, q=1, asymmetric weights Pi_0=(0.8, 0.2).
    S_0^1=80, S_0^2=20; rho_1^1=0.95, rho_1^2=0.10; beta_1=-1.0, beta_2=-0.2.
    """
    rho11 = 0.95
    rho12 = 0.10
    R = np.array([
        [rho11, np.sqrt(1 - rho11**2), 0.0],
        [rho12, 0.0, np.sqrt(1 - rho12**2)],
    ])
    return {
        "name": "Baseline A",
        "n": 2,
        "d": 1,
        "q": 1,
        "S0": np.array([80.0, 20.0]),
        "V0": np.array([0.01, 0.16]),
        "rho_loadings": R,
        "beta": np.array([-1.0, -0.2]),
        "kappa": 1.0,
    }


def baseline_B_config() -> dict:
    """
    Baseline B: n=2, d=1, q=1, symmetric weights Pi_0=(0.5, 0.5).
    S_0^1=S_0^2=50; rho_1^1=rho_1^2=0.70; beta_1=beta_2=-1.0.
    Expected: psi_B = 0 by symmetry.
    """
    rho1 = 0.70
    R = np.array([
        [rho1, np.sqrt(1 - rho1**2), 0.0],
        [rho1, 0.0, np.sqrt(1 - rho1**2)],
    ])
    return {
        "name": "Baseline B",
        "n": 2,
        "d": 1,
        "q": 1,
        "S0": np.array([50.0, 50.0]),
        "V0": np.array([0.04, 0.04]),
        "rho_loadings": R,
        "beta": np.array([-1.0, -1.0]),
        "kappa": 1.0,
    }
