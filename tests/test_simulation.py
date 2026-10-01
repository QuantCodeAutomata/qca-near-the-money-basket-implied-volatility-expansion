"""
Unit tests for src/simulation.py

Tests verify:
- Volterra factor has correct variance (E[Z_t^2] ≈ t^{2H}/(2H))
- Variance paths are positive everywhere
- Martingale condition E[S_theta^i / S_0^i] ≈ 1
- Basket value equals sum of constituents
- Terminal weights sum to 1 on every path
- Antithetic variates produce correct symmetry
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.coefficients import baseline_A_config, baseline_B_config, volterra_variance
from src.simulation import (
    build_volterra_kernel_matrix,
    check_martingale,
    simulate_model_paths,
    simulate_variance_paths,
    simulate_volterra_factor,
)


# ─── Volterra factor ─────────────────────────────────────────────────────────

class TestVolterraFactor:
    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
    def test_variance_close_to_analytic(self, H):
        """E[Z_theta^2] ≈ Var(Z_theta) = theta^{2H} / (2H)."""
        theta = 0.015625  # 2^{-6}
        M = 512
        N = 20000
        dt = theta / M

        rng = np.random.default_rng(42)
        dW1 = np.sqrt(dt) * rng.standard_normal((M, N))
        Z = simulate_volterra_factor(dW1, H, dt)

        # Empirical variance at T
        empirical_var = float(np.var(Z[-1]))
        analytic_var = float(volterra_variance(theta, H))

        rel_err = abs(empirical_var - analytic_var) / analytic_var
        assert rel_err < 0.05, \
            f"H={H}: empirical var {empirical_var:.6f} vs analytic {analytic_var:.6f}, rel_err={rel_err:.3f}"

    def test_initial_value_zero(self):
        """Z_0 = 0 by construction."""
        H = 0.3
        M, N = 64, 100
        dt = 0.01 / M
        rng = np.random.default_rng(0)
        dW1 = np.sqrt(dt) * rng.standard_normal((M, N))
        Z = simulate_volterra_factor(dW1, H, dt)
        assert Z.shape == (M + 1, N)
        np.testing.assert_allclose(Z[0], 0.0, atol=1e-15)

    def test_shape_correctness(self):
        H, M, N = 0.3, 32, 50
        dt = 0.01 / M
        rng = np.random.default_rng(0)
        dW1 = np.sqrt(dt) * rng.standard_normal((M, N))
        Z = simulate_volterra_factor(dW1, H, dt)
        assert Z.shape == (M + 1, N)

    def test_kernel_matrix_lower_triangular(self):
        """Kernel matrix must be lower triangular (no look-ahead in time)."""
        H, M, dt = 0.3, 8, 0.01
        K = build_volterra_kernel_matrix(M, dt, H)
        # Upper triangle (excluding main diagonal and below) should be 0
        assert np.all(np.triu(K, k=1) == 0), "Kernel matrix must be lower triangular"


# ─── Variance paths ──────────────────────────────────────────────────────────

class TestVariancePaths:
    @pytest.mark.parametrize("baseline_fn", [baseline_A_config, baseline_B_config])
    def test_variance_positive_everywhere(self, baseline_fn):
        """V_t^i > 0 on all paths (positivity by exponential construction)."""
        cfg = baseline_fn()
        H = 0.3
        theta = 0.015625
        M, N_half = 64, 500
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=123)
        V = result["V_paths"]
        assert np.all(V > 0), "Variance paths must be strictly positive"

    def test_variance_initial_value(self):
        """V_0^i = V_0^i by construction (Z_0 = 0)."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        M, N_half = 32, 100
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=7)
        V = result["V_paths"]
        V0 = np.asarray(cfg["V0"])
        # V[:, 0, :] should all equal V0[i]
        for i in range(cfg["n"]):
            np.testing.assert_allclose(V[i, 0, :], V0[i], rtol=1e-10)

    def test_expected_variance_near_V0(self):
        """E[V_t^i] = V_0^i exactly in the model (martingale variance)."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        M, N_half = 128, 2000
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=0)
        V = result["V_paths"]
        V0 = np.asarray(cfg["V0"])

        # Check mean at terminal time
        for i in range(cfg["n"]):
            empirical_mean = float(V[i, -1, :].mean())
            rel_err = abs(empirical_mean - V0[i]) / V0[i]
            assert rel_err < 0.10, \
                f"E[V_T^{i}] = {empirical_mean:.6f}, V0[{i}] = {V0[i]:.6f}, rel_err={rel_err:.3f}"


# ─── Asset paths ──────────────────────────────────────────────────────────────

class TestAssetPaths:
    @pytest.mark.parametrize("baseline_fn", [baseline_A_config, baseline_B_config])
    def test_asset_prices_positive(self, baseline_fn):
        """Asset prices must be strictly positive (log-Euler scheme)."""
        cfg = baseline_fn()
        H = 0.3
        theta = 0.015625
        result = simulate_model_paths(cfg, H, theta, M=64, N_half=500, seed=99)
        S = result["S_paths"]
        assert np.all(S > 0), "Asset prices must be positive"

    def test_initial_prices_correct(self):
        """S_0^i must equal the configured S0^i."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        result = simulate_model_paths(cfg, H, theta, M=32, N_half=100, seed=1)
        S = result["S_paths"]
        S0 = np.asarray(cfg["S0"])
        for i in range(cfg["n"]):
            np.testing.assert_allclose(S[i, 0, :], S0[i], rtol=1e-10)

    @pytest.mark.parametrize("baseline_fn", [baseline_A_config, baseline_B_config])
    def test_martingale_condition(self, baseline_fn):
        """E[S_theta^i / S_0^i] ≈ 1 (martingale under Q)."""
        cfg = baseline_fn()
        H = 0.3
        theta = 0.015625
        M, N_half = 128, 3000
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=42)
        S_term = result["S_terminal"]
        S0 = np.asarray(cfg["S0"])
        for i in range(cfg["n"]):
            ratio = S_term[i] / S0[i]
            mean_ratio = float(ratio.mean())
            assert abs(mean_ratio - 1.0) < 0.05, \
                f"Martingale check: E[S_T^{i}/S_0^{i}] = {mean_ratio:.4f}, expected 1.0"


# ─── Basket properties ────────────────────────────────────────────────────────

class TestBasketProperties:
    def test_basket_equals_sum_of_constituents(self):
        """S_basket = sum_i S_terminal^i."""
        cfg = baseline_A_config()
        result = simulate_model_paths(cfg, H=0.3, theta=0.015625, M=64, N_half=500, seed=5)
        S_basket = result["S_basket"]
        S_terminal = result["S_terminal"]
        np.testing.assert_allclose(S_basket, S_terminal.sum(axis=0), rtol=1e-12)

    def test_terminal_weights_sum_to_one(self):
        """Pi_theta^i must sum to 1 on every path."""
        cfg = baseline_A_config()
        result = simulate_model_paths(cfg, H=0.3, theta=0.015625, M=64, N_half=500, seed=3)
        Pi = result["Pi_terminal"]
        np.testing.assert_allclose(Pi.sum(axis=0), 1.0, atol=1e-12)

    def test_terminal_weights_positive(self):
        """Pi_theta^i > 0 on all paths (since S^i > 0)."""
        cfg = baseline_A_config()
        result = simulate_model_paths(cfg, H=0.3, theta=0.015625, M=64, N_half=500, seed=4)
        Pi = result["Pi_terminal"]
        assert np.all(Pi > 0)


# ─── Antithetic variates ─────────────────────────────────────────────────────

class TestAntithetics:
    def test_total_paths_double_N_half(self):
        cfg = baseline_A_config()
        N_half = 500
        result = simulate_model_paths(cfg, H=0.3, theta=0.015625, M=32, N_half=N_half, seed=0)
        assert result["N"] == 2 * N_half

    def test_antithetic_reduces_variance(self):
        """Antithetic estimator should have lower variance for basket payoffs."""
        cfg = baseline_A_config()
        H, theta, M, N_half = 0.3, 0.015625, 64, 1000
        seed = 10

        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=seed)
        S_t = result["S_basket"]
        S0 = float(np.asarray(cfg["S0"]).sum())

        # ATM put payoffs
        payoffs = np.maximum(S0 - S_t, 0.0)
        N = len(S_t)
        half = N // 2

        # Compare variance of original vs antithetic mean
        var_raw = float(payoffs[:half].var(ddof=1))
        var_anti = float(((payoffs[:half] + payoffs[half:]) / 2).var(ddof=1))
        # Antithetic variance should be less than or equal
        assert var_anti <= var_raw * 1.05, \
            f"Antithetic variance {var_anti:.6f} should be ≤ raw variance {var_raw:.6f}"


# ─── Check martingale helper ──────────────────────────────────────────────────

class TestCheckMartingale:
    def test_passes_for_valid_simulation(self):
        cfg = baseline_A_config()
        batches = [
            simulate_model_paths(cfg, H=0.3, theta=0.015625, M=64, N_half=1000, seed=b)
            for b in range(5)
        ]
        result = check_martingale(batches, atol=0.05)
        assert result["passed"], \
            f"Martingale check failed: mean_ratios={result['mean_ratios']}"
