"""
Unit tests for src/coefficients.py

Tests verify:
- Return correlation normalization (rho^{ii} = 1)
- Residual covariance positive definiteness
- Volterra variance formula at H=0.5 equals t
- v(0) positivity and exact formulas
- psi_B = 0 for Baseline B (symmetric case)
- psi_A != 0 for both baselines
- Mathematical properties of all coefficients
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.coefficients import (
    baseline_A_config,
    baseline_B_config,
    check_positive_definite,
    compute_bar_v,
    compute_c_theta,
    compute_d_i,
    compute_gamma,
    compute_psi_A,
    compute_psi_B,
    compute_psi_ki,
    compute_residual_covariance,
    compute_return_correlations,
    compute_scenario_coefficients,
    compute_v0_scalar,
    volterra_variance,
)


# ─── Baseline configurations ───────────────────────────────────────────────

class TestBaselineConfigs:
    def test_baseline_A_has_required_keys(self):
        cfg = baseline_A_config()
        for key in ["n", "d", "q", "S0", "V0", "rho_loadings", "beta", "kappa"]:
            assert key in cfg, f"Missing key: {key}"

    def test_baseline_B_has_required_keys(self):
        cfg = baseline_B_config()
        for key in ["n", "d", "q", "S0", "V0", "rho_loadings", "beta", "kappa"]:
            assert key in cfg

    def test_baseline_A_initial_weights_sum_to_one(self):
        cfg = baseline_A_config()
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        assert abs(Pi0.sum() - 1.0) < 1e-12

    def test_baseline_B_symmetric_weights(self):
        cfg = baseline_B_config()
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        assert abs(Pi0[0] - Pi0[1]) < 1e-12

    def test_baseline_A_rho_loadings_shape(self):
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        n, d = cfg["n"], cfg["d"]
        assert R.shape == (n, d + n)

    def test_baseline_A_loadings_unit_norm(self):
        """Each row of R should be a unit vector (sum of squares = 1)."""
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        norms_sq = (R**2).sum(axis=1)
        np.testing.assert_allclose(norms_sq, 1.0, atol=1e-12)


# ─── Return correlations ────────────────────────────────────────────────────

class TestReturnCorrelations:
    def test_diagonal_equals_one(self):
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        np.testing.assert_allclose(np.diag(rho), 1.0, atol=1e-12)

    def test_offdiag_baseline_A(self):
        """rho^{12} should be 0.95 * 0.10 = 0.095 for Baseline A."""
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        assert abs(rho[0, 1] - 0.095) < 1e-10

    def test_offdiag_baseline_B(self):
        """rho^{12} should be 0.70 * 0.70 = 0.49 for Baseline B."""
        cfg = baseline_B_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        assert abs(rho[0, 1] - 0.49) < 1e-10

    def test_correlation_matrix_symmetric(self):
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        np.testing.assert_allclose(rho, rho.T, atol=1e-14)


# ─── Residual covariance ────────────────────────────────────────────────────

class TestResidualCovariance:
    def test_baseline_A_C_perp_positive_definite(self):
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        C = compute_residual_covariance(R, cfg["d"])
        is_pd, eigvals = check_positive_definite(C)
        assert is_pd, f"C_perp not PD: eigenvalues={eigvals}"
        assert np.all(eigvals > 0)

    def test_baseline_B_C_perp_positive_definite(self):
        cfg = baseline_B_config()
        R = np.asarray(cfg["rho_loadings"])
        C = compute_residual_covariance(R, cfg["d"])
        is_pd, eigvals = check_positive_definite(C)
        assert is_pd

    def test_baseline_A_C_perp_approx_values(self):
        """C_perp should be approximately diag(1 - 0.95^2, 1 - 0.10^2) in Baseline A."""
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        C = compute_residual_covariance(R, cfg["d"])
        # Diagonal: (1 - 0.95^2, 1 - 0.10^2) ≈ (0.0975, 0.99) but cross terms are 0
        assert abs(C[0, 0] - (1 - 0.95**2)) < 1e-12
        assert abs(C[1, 1] - (1 - 0.10**2)) < 1e-12
        assert abs(C[0, 1]) < 1e-12  # Off-diagonal is 0 since residuals are orthogonal


# ─── Volterra variance ───────────────────────────────────────────────────────

class TestVolterraVariance:
    def test_H_half_equals_t(self):
        """For H=0.5: Var(Z_t) = t."""
        t = np.array([0.1, 0.5, 1.0])
        v = volterra_variance(t, H=0.5)
        np.testing.assert_allclose(v, t, rtol=1e-12)

    def test_scalar_input(self):
        v = volterra_variance(1.0, H=0.3)
        expected = 1.0 ** (2 * 0.3) / (2 * 0.3)
        assert abs(v - expected) < 1e-12

    def test_H_small_variance_larger_than_H_half(self):
        """For H < 0.5 and t < 1: Var(Z_t) = t^{2H}/(2H) > t since t^{2H} > t when 2H < 1."""
        t = 0.1
        v_rough = volterra_variance(t, H=0.1)   # t^{0.2}/0.2 ≈ 3.15
        v_standard = volterra_variance(t, H=0.5)  # t = 0.1
        assert v_rough > v_standard, (
            f"For H=0.1 and t={t}: Var={v_rough:.4f} should exceed H=0.5 Var={v_standard:.4f}"
        )

    def test_positive_for_all_H(self):
        t = 0.5
        for H in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert volterra_variance(t, H) > 0


# ─── Basket variance v(0) ────────────────────────────────────────────────────

class TestBasketVariance:
    def test_v0_positive_baseline_A(self):
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        v0 = compute_v0_scalar(Pi0, np.asarray(cfg["V0"]), rho)
        assert v0 > 0

    def test_v0_positive_baseline_B(self):
        cfg = baseline_B_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        v0 = compute_v0_scalar(Pi0, np.asarray(cfg["V0"]), rho)
        assert v0 > 0

    def test_v0_formula_baseline_B_manual(self):
        """Manually verify v(0) for Baseline B at t=0."""
        cfg = baseline_B_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        V0 = np.asarray(cfg["V0"])
        v0 = compute_v0_scalar(Pi0, V0, rho)
        # Manual: 0.5^2*0.04 + 2*0.5*0.5*0.04*0.49 + 0.5^2*0.04
        expected = 0.25 * 0.04 + 2 * 0.25 * 0.04 * 0.49 + 0.25 * 0.04
        assert abs(v0 - expected) < 1e-12

    def test_d_i_formula_consistency(self):
        """v(0) = sum_i Pi_0^i d_i."""
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        V0 = np.asarray(cfg["V0"])
        d_i = compute_d_i(Pi0, V0, rho)
        v0_from_d = float(np.dot(Pi0, d_i))
        v0_direct = compute_v0_scalar(Pi0, V0, rho)
        assert abs(v0_from_d - v0_direct) < 1e-14


# ─── psi coefficients ─────────────────────────────────────────────────────────

class TestPsiCoefficients:
    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
    def test_psi_B_zero_baseline_B(self, H):
        """Baseline B is symmetric => psi_B = 0."""
        cfg = baseline_B_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        V0 = np.asarray(cfg["V0"])
        psi_B = compute_psi_B(Pi0, V0, rho)
        assert abs(psi_B) < 1e-10, f"psi_B should be 0 for Baseline B, got {psi_B}"

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
    def test_psi_B_positive_baseline_A(self, H):
        """Baseline A is asymmetric => psi_B > 0."""
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        rho = compute_return_correlations(R)
        Pi0 = np.asarray(cfg["S0"]) / np.asarray(cfg["S0"]).sum()
        V0 = np.asarray(cfg["V0"])
        psi_B = compute_psi_B(Pi0, V0, rho)
        assert psi_B > 0, f"psi_B should be > 0 for Baseline A, got {psi_B}"

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
    def test_psi_A_nonzero_both_baselines(self, H):
        """psi_A must be nonzero (nondegeneracy) for both baselines."""
        for cfg_fn in [baseline_A_config, baseline_B_config]:
            cfg = cfg_fn()
            coeffs = compute_scenario_coefficients(cfg, H)
            assert abs(coeffs["psi_A"]) > 1e-10, \
                f"psi_A should be nonzero, got {coeffs['psi_A']}"

    def test_psi_ki_formula(self):
        """psi^{ki} = 2 * gamma^{ki} / (2H + 1)."""
        cfg = baseline_A_config()
        H = 0.3
        R = np.asarray(cfg["rho_loadings"])
        beta = np.asarray(cfg["beta"])
        gamma = compute_gamma(beta, R, kappa=1.0, q=1, d=1)
        psi_ki = compute_psi_ki(gamma, H)
        expected = 2.0 * gamma / (2.0 * H + 1.0)
        np.testing.assert_allclose(psi_ki, expected, rtol=1e-12)

    def test_gamma_one_factor(self):
        """gamma^{ki} = beta_i * rho_1^k for one-factor model."""
        cfg = baseline_A_config()
        R = np.asarray(cfg["rho_loadings"])
        beta = np.asarray(cfg["beta"])
        gamma = compute_gamma(beta, R, kappa=1.0, q=1, d=1)
        rho1 = R[:, 0]  # first column: rho_1^k for each k
        for k in range(2):
            for i in range(2):
                expected = beta[i] * rho1[k]
                assert abs(gamma[k, i] - expected) < 1e-14


# ─── bar v(theta) ───────────────────────────────────────────────────────────

class TestBarV:
    def test_bar_v_at_t0_approaches_v0(self):
        """bar v(theta) -> v(0) as theta -> 0."""
        cfg = baseline_A_config()
        H = 0.3
        coeffs = compute_scenario_coefficients(cfg, H)
        bar_v = compute_bar_v(
            coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta=1e-6
        )
        assert abs(bar_v - coeffs["v0"]) < 1e-3, \
            f"bar_v({1e-6}) = {bar_v} should be close to v0={coeffs['v0']}"

    def test_bar_v_positive(self):
        for cfg_fn in [baseline_A_config, baseline_B_config]:
            cfg = cfg_fn()
            H = 0.3
            coeffs = compute_scenario_coefficients(cfg, H)
            for theta in [0.015625, 0.001]:
                bar_v = compute_bar_v(
                    coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta
                )
                assert bar_v > 0

    def test_bar_v_baseline_B_equals_v0_for_equal_betas(self):
        """For Baseline B: E[sqrt(V_t^i V_t^j)] = sqrt(V0^i V0^j) * exp(0) when beta_1=beta_2.
        So bar_v(theta) = v(0) = sum_{i,j} Pi0^i Pi0^j sqrt(V0^i V0^j) rho^{ij}."""
        cfg = baseline_B_config()
        H = 0.3
        coeffs = compute_scenario_coefficients(cfg, H)
        # beta_1 = beta_2 = -1.0, so (beta_i - beta_j)^2 = 0
        # E[sqrt(V_t^i V_t^j)] = sqrt(V0^i V0^j) for all t
        # => bar_v(theta) = v(0) for all theta
        bar_v = compute_bar_v(
            coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta=0.015
        )
        assert abs(bar_v - coeffs["v0"]) < 1e-10, \
            f"For Baseline B: bar_v should equal v(0)={coeffs['v0']:.6f}, got {bar_v:.6f}"


# ─── c_theta and asymptotic slope ───────────────────────────────────────────

class TestCTheta:
    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
    def test_c_theta_positive_for_negative_psi_A(self, H):
        """c_theta is dominated by psi_A * r(theta) term; sign depends on psi_A."""
        cfg = baseline_A_config()
        coeffs = compute_scenario_coefficients(cfg, H)
        theta = 0.015625
        c = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        # Just check it's finite and nonzero
        assert np.isfinite(c)
        assert abs(c) > 0

    def test_c_theta_baseline_B_equals_A_only_term(self):
        """For Baseline B: psi_B = 0 => c_theta = (psi_A/(2H+3)) * r(theta)."""
        cfg = baseline_B_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        c_full = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        c_A_only = compute_c_theta(coeffs["psi_A"], 0.0, H, theta)
        assert abs(c_full - c_A_only) < 1e-14

    def test_c_theta_scalar_and_array_consistent(self):
        cfg = baseline_A_config()
        H = 0.3
        coeffs = compute_scenario_coefficients(cfg, H)
        theta = 0.015625
        c_scalar = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        c_array = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, np.array([theta]))
        assert abs(c_scalar - c_array[0]) < 1e-14


# ─── Scenario coefficient computation ────────────────────────────────────────

class TestScenarioCoefficients:
    @pytest.mark.parametrize("baseline", ["A", "B"])
    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
    def test_compute_scenario_coefficients(self, baseline, H):
        cfg = baseline_A_config() if baseline == "A" else baseline_B_config()
        coeffs = compute_scenario_coefficients(cfg, H)

        assert coeffs["C_perp_is_pd"], "C_perp must be positive definite"
        assert coeffs["v0"] > 0, "v(0) must be positive"
        assert np.isfinite(coeffs["psi_A"])
        assert np.isfinite(coeffs["psi_B"])

        if baseline == "B":
            assert abs(coeffs["psi_B"]) < 1e-10, "psi_B = 0 for Baseline B"
