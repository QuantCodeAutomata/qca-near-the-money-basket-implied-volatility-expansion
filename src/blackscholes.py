"""
Black-Scholes put pricing and implied volatility inversion.

Implements the paper's normalized BS put formula:
  p(k, w) = exp(k) * Phi(-d_-(k, w)) - Phi(-d_+(k, w))
  d_pm = -k/w ± w/2

where k = log(K/S0) is log-moneyness and w = sigma * sqrt(theta) is total vol.

Custom — Context7 found no library equivalent for this normalized form.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_Phi = norm.cdf
_phi = norm.pdf


def bs_d_plus(k: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
    """d_+ = -k/w + w/2"""
    return -k / w + w / 2.0


def bs_d_minus(k: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
    """d_- = -k/w - w/2"""
    return -k / w - w / 2.0


def bs_put_normalized(k: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
    """
    Normalized Black-Scholes put price (divided by S_0):
    p(k, w) = exp(k) * Phi(-d_-(k, w)) - Phi(-d_+(k, w)).

    Parameters
    ----------
    k : log-moneyness = log(K/S0)
    w : total vol = sigma * sqrt(theta)

    Returns
    -------
    p : float or ndarray
    """
    d_plus = bs_d_plus(k, w)
    d_minus = bs_d_minus(k, w)
    return np.exp(k) * _Phi(-d_minus) - _Phi(-d_plus)


def bs_put_price(
    S0: float,
    K: float | np.ndarray,
    sigma: float | np.ndarray,
    theta: float,
) -> float | np.ndarray:
    """
    Black-Scholes put price (no discounting, risk-neutral with zero rate).
    P(K, theta) = S0 * p(k, w).

    Parameters
    ----------
    S0 : float
    K : float or ndarray
    sigma : float or ndarray - implied volatility
    theta : float - maturity

    Returns
    -------
    price : float or ndarray
    """
    k = np.log(K / S0)
    w = sigma * np.sqrt(theta)
    return S0 * bs_put_normalized(k, w)


def implied_vol_from_price(
    price: float,
    S0: float,
    K: float,
    theta: float,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
) -> float:
    """
    Invert Black-Scholes put price to get implied volatility using Brent's method.

    Parameters
    ----------
    price : float
        Monte Carlo put price.
    S0 : float
    K : float
    theta : float
    sigma_lo, sigma_hi : float
        Bracket for implied vol search.

    Returns
    -------
    sigma_imp : float
        Implied volatility, or NaN if inversion fails.
    """
    k = np.log(K / S0)

    # Intrinsic value check
    intrinsic = max(K - S0, 0.0)
    if price <= intrinsic + 1e-12:
        return np.nan

    def objective(sigma: float) -> float:
        return bs_put_price(S0, K, sigma, theta) - price

    try:
        lo = objective(sigma_lo)
        hi = objective(sigma_hi)
        if lo * hi > 0:
            return np.nan
        return brentq(objective, sigma_lo, sigma_hi, xtol=1e-10, rtol=1e-10)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_vectorized(
    prices: np.ndarray,
    S0: float,
    K_arr: np.ndarray,
    theta: float,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
) -> np.ndarray:
    """
    Vectorized implied vol inversion over an array of prices and strikes.

    Parameters
    ----------
    prices : ndarray of shape (m,)
    S0 : float
    K_arr : ndarray of shape (m,)
    theta : float
    sigma_lo, sigma_hi : float

    Returns
    -------
    sigmas : ndarray of shape (m,) with NaN for failures
    """
    sigmas = np.full(len(prices), np.nan)
    for i, (p, K) in enumerate(zip(prices, K_arr)):
        sigmas[i] = implied_vol_from_price(p, S0, K, theta, sigma_lo, sigma_hi)
    return sigmas


def atm_put_price_from_mc(
    S_basket_terminal: np.ndarray,
    S0_basket: float,
) -> tuple[float, float]:
    """
    Estimate ATM put price E[(S0 - S_theta)_+] from Monte Carlo paths.

    Parameters
    ----------
    S_basket_terminal : ndarray of shape (N,)
    S0_basket : float

    Returns
    -------
    price_mean : float
    price_se : float - standard error
    """
    payoffs = np.maximum(S0_basket - S_basket_terminal, 0.0)
    return float(payoffs.mean()), float(payoffs.std(ddof=1) / np.sqrt(len(payoffs)))


def put_price_from_mc(
    S_basket_terminal: np.ndarray,
    K: float,
) -> tuple[float, float]:
    """
    Estimate put price E[(K - S_theta)_+] from Monte Carlo paths.

    Parameters
    ----------
    S_basket_terminal : ndarray of shape (N,)
    K : float

    Returns
    -------
    price_mean : float
    price_se : float
    """
    payoffs = np.maximum(K - S_basket_terminal, 0.0)
    return float(payoffs.mean()), float(payoffs.std(ddof=1) / np.sqrt(len(payoffs)))


def digital_probability(
    S_basket_terminal: np.ndarray,
    S0_basket: float,
) -> tuple[float, float]:
    """
    Estimate Q(S_theta < S_0) from Monte Carlo indicator.

    Parameters
    ----------
    S_basket_terminal : ndarray of shape (N,)
    S0_basket : float

    Returns
    -------
    prob : float
    se : float
    """
    indicator = (S_basket_terminal < S0_basket).astype(float)
    N = len(indicator)
    prob = float(indicator.mean())
    se = float(indicator.std(ddof=1) / np.sqrt(N))
    return prob, se


def atm_skew_from_digital(
    Q_digital: float,
    sigma_atm: float,
    theta: float,
) -> float:
    """
    Paper-faithful ATM skew estimator using the digital identity:
    partial_k sigma(0, theta) = [Q(S_theta < S_0) - Phi(sqrt(theta) sigma_atm / 2)]
                                / [phi(sqrt(theta) sigma_atm / 2) * sqrt(theta)]

    Parameters
    ----------
    Q_digital : float
        Estimated Q(S_theta < S_0).
    sigma_atm : float
        ATM implied volatility sigma(0, theta).
    theta : float

    Returns
    -------
    skew : float
    """
    w_atm = np.sqrt(theta) * sigma_atm / 2.0
    numerator = Q_digital - _Phi(w_atm)
    denominator = _phi(w_atm) * np.sqrt(theta)
    if abs(denominator) < 1e-15:
        return np.nan
    return numerator / denominator


def atm_implied_vol(
    S_basket_terminal: np.ndarray,
    S0_basket: float,
    theta: float,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
) -> float:
    """
    Compute ATM implied volatility from MC put prices.

    Parameters
    ----------
    S_basket_terminal : ndarray (N,)
    S0_basket : float
    theta : float

    Returns
    -------
    sigma_atm : float
    """
    price, _ = atm_put_price_from_mc(S_basket_terminal, S0_basket)
    return implied_vol_from_price(price, S0_basket, S0_basket, theta, sigma_lo, sigma_hi)
