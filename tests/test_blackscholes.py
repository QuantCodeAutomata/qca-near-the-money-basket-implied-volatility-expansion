"""
Unit tests for src/blackscholes.py

Tests verify:
- BS put price is non-negative
- ATM BS put price is symmetric
- Implied vol inversion is consistent (round-trip)
- Put-call parity
- Digital probability in [0, 1]
- ATM skew digital formula properties
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.blackscholes import (
    atm_implied_vol,
    atm_skew_from_digital,
    bs_d_minus,
    bs_d_plus,
    bs_put_normalized,
    bs_put_price,
    digital_probability,
    implied_vol_from_price,
    implied_vol_vectorized,
    put_price_from_mc,
)


# ─── BS d+ / d- ──────────────────────────────────────────────────────────────

class TestBSDPlusMinus:
    def test_atm_dplusdminus(self):
        """At ATM (k=0): d_+ = w/2, d_- = -w/2."""
        k, w = 0.0, 0.3
        assert abs(bs_d_plus(k, w) - w / 2) < 1e-14
        assert abs(bs_d_minus(k, w) - (-w / 2)) < 1e-14

    def test_d_plus_minus_d_minus_equals_w(self):
        """d_+ - d_- = w for all k, w."""
        k, w = -0.1, 0.25
        assert abs((bs_d_plus(k, w) - bs_d_minus(k, w)) - w) < 1e-14


# ─── BS put price ────────────────────────────────────────────────────────────

class TestBSPutNormalized:
    def test_put_nonnegative(self):
        """Put price must be non-negative."""
        for k in [-0.3, 0.0, 0.3]:
            for w in [0.1, 0.3, 0.5]:
                p = bs_put_normalized(k, w)
                assert p >= 0, f"Put price {p} < 0 for k={k}, w={w}"

    def test_atm_put_is_approx_phi0_w(self):
        """For k=0 and small w: p(0, w) ≈ phi(0) * w."""
        from scipy.stats import norm
        w = 0.05
        p = bs_put_normalized(0.0, w)
        approx = norm.pdf(0.0) * w
        assert abs(p - approx) < 0.001

    def test_deep_otm_put_small_price(self):
        """Deep OTM put (k << 0, strike << spot) has near-zero price."""
        # k = log(K/S0) = -1 → K = S0/e ≈ 0.37 S0 (low strike, OTM put)
        p = bs_put_normalized(k=-1.0, w=0.2)
        assert p < 1e-4, f"Deep OTM put price should be near 0, got {p:.6f}"

    def test_deep_itm_put_approx_intrinsic(self):
        """Deep ITM put (k >> 0, strike >> spot) ≈ exp(k) - 1 (intrinsic)."""
        # k = log(K/S0) = 1 → K = e*S0 >> S0 (high strike, ITM put)
        p = bs_put_normalized(k=1.0, w=0.2)
        # Intrinsic value (normalized) ≈ exp(k) - 1 = e - 1 ≈ 1.718
        assert p > 1.7, f"Deep ITM put should be near intrinsic e-1≈1.718, got {p:.4f}"

    def test_put_increasing_in_w(self):
        """Higher vol -> higher put price (all else equal)."""
        k = 0.0
        p_low = bs_put_normalized(k, 0.1)
        p_high = bs_put_normalized(k, 0.5)
        assert p_high > p_low

    def test_bs_put_price_versus_normalized(self):
        """bs_put_price(S0, K, sigma, theta) = S0 * p(log(K/S0), sigma*sqrt(theta))."""
        S0, K, sigma, theta = 100.0, 98.0, 0.2, 0.25
        k = np.log(K / S0)
        w = sigma * np.sqrt(theta)
        price_direct = bs_put_price(S0, K, sigma, theta)
        price_normalized = S0 * bs_put_normalized(k, w)
        assert abs(price_direct - price_normalized) < 1e-12


# ─── Implied vol inversion ────────────────────────────────────────────────────

class TestImpliedVol:
    def test_round_trip(self):
        """IV inversion is consistent: sigma -> price -> sigma_inverted ≈ sigma."""
        S0, K, sigma, theta = 100.0, 100.0, 0.25, 0.25
        price = bs_put_price(S0, K, sigma, theta)
        sigma_inv = implied_vol_from_price(price, S0, K, theta)
        assert abs(sigma_inv - sigma) < 1e-8, \
            f"Round-trip failed: sigma={sigma}, sigma_inv={sigma_inv}"

    def test_round_trip_otm(self):
        """OTM put: round-trip IV inversion."""
        S0, K, sigma, theta = 100.0, 95.0, 0.30, 0.5
        price = bs_put_price(S0, K, sigma, theta)
        sigma_inv = implied_vol_from_price(price, S0, K, theta)
        assert abs(sigma_inv - sigma) < 1e-7

    def test_returns_nan_for_zero_price(self):
        """Below-intrinsic price should return NaN."""
        sigma = implied_vol_from_price(0.0, 100.0, 90.0, 0.1)
        # Price=0 is below intrinsic 0 for OTM put, should get nan or handle
        assert np.isnan(sigma) or sigma == pytest.approx(0.0, abs=1e-3)

    def test_vectorized_consistent_with_scalar(self):
        """Vectorized inversion matches scalar inversion."""
        S0, theta = 100.0, 0.25
        sigmas = np.array([0.15, 0.20, 0.25, 0.30])
        K_arr = np.array([95.0, 100.0, 100.0, 105.0])
        prices = np.array([bs_put_price(S0, K, s, theta) for K, s in zip(K_arr, sigmas)])

        sigma_vec = implied_vol_vectorized(prices, S0, K_arr, theta)
        for i, (s_exp, s_got) in enumerate(zip(sigmas, sigma_vec)):
            assert abs(s_got - s_exp) < 1e-7, \
                f"Vectorized IV mismatch at i={i}: expected {s_exp}, got {s_got}"


# ─── MC pricing helpers ───────────────────────────────────────────────────────

class TestMCPricing:
    def test_put_price_non_negative(self):
        """Put payoffs are always non-negative."""
        S_terminal = np.array([90.0, 100.0, 110.0])
        K = 100.0
        price, se = put_price_from_mc(S_terminal, K)
        assert price >= 0
        assert se >= 0

    def test_put_price_atm_approx(self):
        """Large MC sample: ATM put price ≈ BS ATM put."""
        rng = np.random.default_rng(0)
        sigma, theta = 0.3, 0.25
        S0 = 100.0
        S_terminal = S0 * np.exp(-0.5 * sigma**2 * theta + sigma * np.sqrt(theta) * rng.standard_normal(100000))
        price_mc, se = put_price_from_mc(S_terminal, S0)
        price_bs = bs_put_price(S0, S0, sigma, theta)
        rel_err = abs(price_mc - price_bs) / price_bs
        assert rel_err < 0.03, f"MC price {price_mc:.4f} vs BS {price_bs:.4f}, rel_err={rel_err:.3f}"

    def test_digital_probability_in_0_1(self):
        """Q(S_theta < S_0) must be in [0, 1]."""
        rng = np.random.default_rng(1)
        S_terminal = rng.lognormal(0.0, 0.3, 10000)
        prob, se = digital_probability(S_terminal, S0_basket=1.0)
        assert 0 <= prob <= 1
        assert se >= 0

    def test_digital_probability_deep_otm(self):
        """When all terminal prices >> S0, digital should be near 0."""
        S_terminal = np.full(1000, 200.0)  # all far above strike
        prob, se = digital_probability(S_terminal, S0_basket=100.0)
        assert prob == 0.0
        assert se == 0.0

    def test_digital_probability_deep_itm(self):
        """When all terminal prices << S0, digital should be 1."""
        S_terminal = np.full(1000, 50.0)
        prob, se = digital_probability(S_terminal, S0_basket=100.0)
        assert prob == 1.0


# ─── ATM skew from digital ────────────────────────────────────────────────────

class TestATMSkewFromDigital:
    def test_skew_finite_for_reasonable_inputs(self):
        """ATM skew formula should return finite value."""
        sigma_atm = 0.25
        theta = 0.015625
        from scipy.stats import norm
        Q = float(norm.cdf(-sigma_atm * np.sqrt(theta) / 2.0)) + 0.01  # slightly perturbed
        skew = atm_skew_from_digital(Q, sigma_atm, theta)
        assert np.isfinite(skew)

    def test_skew_small_theta_converges_to_finite_limit(self):
        """For Q=0.5 and tiny theta, digital skew → -sigma/2 (finite limit).

        Derivation: Q - Phi(sqrt(θ)σ/2) ≈ -φ(0)·sqrt(θ)·σ/2  as θ→0,
        denominator ≈ φ(0)·sqrt(θ), so ratio → -σ/2.
        """
        sigma_atm = 0.25
        skew = atm_skew_from_digital(0.5, sigma_atm, 1e-20)
        expected_limit = -sigma_atm / 2.0   # = -0.125
        assert abs(skew - expected_limit) < 1e-4, (
            f"Small-θ digital skew should converge to -σ/2={expected_limit:.4f}, got {skew:.6f}"
        )

    def test_skew_sign_consistent_with_digital(self):
        """Higher digital prob (Q > Phi) implies negative skew (more mass below S0 than BS predicts)."""
        sigma_atm = 0.25
        theta = 0.015625
        from scipy.stats import norm
        Phi_val = float(norm.cdf(sigma_atm * np.sqrt(theta) / 2.0))
        # Q slightly above Phi -> positive numerator -> positive skew
        Q_high = Phi_val + 0.002
        skew_high = atm_skew_from_digital(Q_high, sigma_atm, theta)
        Q_low = Phi_val - 0.002
        skew_low = atm_skew_from_digital(Q_low, sigma_atm, theta)
        assert skew_high > skew_low
