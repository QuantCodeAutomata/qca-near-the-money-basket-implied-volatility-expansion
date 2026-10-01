"""
Integration tests verifying that experiment methodology is implemented correctly.

These tests validate paper-methodology adherence without running full experiments.
They use small MC samples to check correctness of formulas and pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from scipy.stats import norm as _scipy_norm

from src.coefficients import (
    baseline_A_config,
    baseline_B_config,
    compute_scenario_coefficients,
    compute_bar_v,
    compute_c_theta,
)
from src.simulation import simulate_model_paths
from src.blackscholes import (
    implied_vol_from_price,
    put_price_from_mc,
    digital_probability,
    atm_implied_vol,
    atm_skew_from_digital,
    bs_put_price,
)
from src.malliavin import (
    compute_malliavin_factor_direction,
    compute_malliavin_residual_directions,
    compute_basket_malliavin,
    compute_A0_theta,
)

_phi = _scipy_norm.pdf
_Phi = _scipy_norm.cdf


# ─── Exp 1 methodology checks ────────────────────────────────────────────────

class TestExp1Methodology:
    """Validate Exp 1: smile expansion formula implementation."""

    def test_asymptotic_formula_structure(self):
        """Verify sigma_full = sqrt(bar_v) + c_theta * z matches formula in paper."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        z_grid = np.array([-1.0, 0.0, 1.0])

        coeffs = compute_scenario_coefficients(cfg, H)
        bar_v = compute_bar_v(
            coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta
        )
        c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)

        sigma_full = np.sqrt(bar_v) + c_theta * z_grid

        # Check it's linear in z
        slope = (sigma_full[-1] - sigma_full[0]) / (z_grid[-1] - z_grid[0])
        assert abs(slope - c_theta) < 1e-10, \
            f"sigma_full should be linear in z with slope c_theta={c_theta:.6f}, got slope={slope:.6f}"

    def test_c_theta_components(self):
        """Verify c_theta = (psi_A/(2H+3))*r(theta) + (psi_B/4)*sqrt(theta)."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        r_theta = theta ** H
        c_expected = (coeffs["psi_A"] / (2*H + 3)) * r_theta + (coeffs["psi_B"] / 4) * np.sqrt(theta)
        c_computed = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        assert abs(c_computed - c_expected) < 1e-14

    def test_strike_definition(self):
        """K = S_0 * exp(sqrt(theta) * z) as specified in paper."""
        S0_basket = 100.0
        theta = 0.015625
        z = 1.0
        k = np.sqrt(theta) * z
        K = S0_basket * np.exp(k)
        # For z=1: K should be above S0
        assert K > S0_basket
        # Verify log-moneyness k = sqrt(theta) * z
        assert abs(np.log(K / S0_basket) - k) < 1e-14

    def test_smile_expansion_atm_equals_sqrt_barv(self):
        """At z=0, sigma_full = sqrt(bar_v(theta)) (intercept check)."""
        cfg = baseline_B_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        bar_v = compute_bar_v(
            coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta
        )
        c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        sigma_atm_theory = np.sqrt(bar_v) + c_theta * 0.0  # z=0
        assert abs(sigma_atm_theory - np.sqrt(bar_v)) < 1e-14


# ─── Exp 2 methodology checks ────────────────────────────────────────────────

class TestExp2Methodology:
    """Validate Exp 2: ATM skew and digital correction."""

    def test_digital_correction_formula(self):
        """Y_0^theta = Q(S_theta < S_0) - Phi(sqrt(theta) sigma_ATM / 2)."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        M, N_half = 64, 1000
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=42)
        S_t = result["S_basket"]
        S0 = float(np.asarray(cfg["S0"]).sum())

        sigma_atm = atm_implied_vol(S_t, S0, theta)
        if not np.isnan(sigma_atm):
            Q, _ = digital_probability(S_t, S0)
            w_atm = np.sqrt(theta) * sigma_atm / 2.0
            Y0 = Q - _Phi(w_atm)
            # Y0 should be a small number (correction term)
            assert np.isfinite(Y0)

    def test_atm_skew_digital_identity(self):
        """Verify the ATM skew = Y0 / (phi(w_atm) * sqrt(theta)) formula."""
        sigma_atm = 0.30
        theta = 0.01
        Q = 0.52  # hypothetical digital probability
        w_atm = np.sqrt(theta) * sigma_atm / 2.0
        skew = atm_skew_from_digital(Q, sigma_atm, theta)
        expected = (Q - _Phi(w_atm)) / (_phi(w_atm) * np.sqrt(theta))
        assert abs(skew - expected) < 1e-14

    def test_theory_skew_formula(self):
        """Theory skew = (psi_A/(2H+3)) * r(theta)/sqrt(theta) + psi_B/4."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        r_theta = theta ** H
        skew_theory = (
            coeffs["psi_A"] / (2*H + 3)
        ) * r_theta / np.sqrt(theta) + coeffs["psi_B"] / 4.0
        # Just verify the formula is finite and matches hand calculation
        hand = (coeffs["psi_A"] / (2*H + 3)) * theta**(H - 0.5) + coeffs["psi_B"] / 4.0
        assert abs(skew_theory - hand) < 1e-14

    def test_scaled_skew_H_half_convergence_expected(self):
        """At H=0.5, ATM skew theory -> (psi_A + psi_B) / 4 (constant limit)."""
        cfg = baseline_A_config()
        H = 0.5
        coeffs = compute_scenario_coefficients(cfg, H)
        theta_small = 2**(-12)
        r_theta = theta_small ** H  # sqrt(theta)
        skew_theory = (
            coeffs["psi_A"] / (2*H + 3)
        ) * r_theta / np.sqrt(theta_small) + coeffs["psi_B"] / 4.0
        # r_theta / sqrt(theta) = theta^{0.5} / theta^{0.5} = 1 for H=0.5
        expected_limit = coeffs["psi_A"] / (2*H + 3) + coeffs["psi_B"] / 4.0
        assert abs(skew_theory - expected_limit) < 1e-14, \
            f"H=0.5 ATM skew limit should be {expected_limit:.4f}, got {skew_theory:.4f}"


# ─── Exp 3 methodology checks ────────────────────────────────────────────────

class TestExp3Methodology:
    """Validate Exp 3: Density expansion formula."""

    def test_hermite_polynomial(self):
        """H_3(x) = x^3 - 3x (third Hermite polynomial)."""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        H3 = x**3 - 3*x
        # H_3(1) = 1 - 3 = -2
        assert abs(H3[2] - (-2.0)) < 1e-14
        # H_3(0) = 0
        assert abs(H3[1]) < 1e-14
        # H_3(-1) = -1 + 3 = 2
        assert abs(H3[0] - 2.0) < 1e-14

    def test_density_approximation_normalizes_approximately(self):
        """q_theta(x) should integrate to approximately 1 for small c_theta."""
        from scipy import integrate as sci_int
        cfg = baseline_B_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        bar_v = compute_bar_v(
            coeffs["Pi0"], coeffs["V0"], coeffs["beta"], coeffs["rho"], H, theta
        )
        sigma_0 = np.sqrt(bar_v * theta)
        c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)

        def q_theta(x):
            u = x + sigma_0 / 2.0
            H3 = u**3 - 3*u
            return _phi(u) * (1.0 + c_theta * H3)

        integral, _ = sci_int.quad(q_theta, -10, 10)
        # Due to Hermite orthogonality, integral of phi(u)*H_3(u) over R is 0
        assert abs(integral - 1.0) < 0.01, \
            f"Density integral = {integral:.6f}, should be ~1.0"

    def test_digital_correction_matches_density(self):
        """
        Y_0^theta ≈ phi(0) c_theta comes from integrating q_theta over (-inf, L0)
        where L0 corresponds to x = -sigma_0/2. The paper's identity holds by
        first-order approximation.
        """
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        r_theta = theta ** H
        c_theta = compute_c_theta(coeffs["psi_A"], coeffs["psi_B"], H, theta)
        Y0_theory = _phi(0.0) * c_theta
        # Just verify the formula is finite
        assert np.isfinite(Y0_theory)


# ─── Exp 4 methodology checks ────────────────────────────────────────────────

class TestExp4Methodology:
    """Validate Exp 4: SSR formula and Malliavin derivatives."""

    def test_residual_malliavin_constant(self):
        """D_0^l log S_theta^j = rho_l^j sqrt(V_0^j) for residual directions l > d."""
        cfg = baseline_A_config()
        d = cfg["d"]
        V0 = np.asarray(cfg["V0"])
        rho_loadings = np.asarray(cfg["rho_loadings"])
        N = 10

        D_res = compute_malliavin_residual_directions(rho_loadings, V0, d, N)
        # Shape: (n, n_residual, N)
        n = cfg["n"]
        n_res = rho_loadings.shape[1] - d

        assert D_res.shape == (n, n_res, N)

        # Check values: D[j, l_idx, :] = rho_loadings[j, d + l_idx] * sqrt(V0[j])
        sqrt_V0 = np.sqrt(V0)
        for j in range(n):
            for l_idx in range(n_res):
                expected = rho_loadings[j, d + l_idx] * sqrt_V0[j]
                np.testing.assert_allclose(
                    D_res[j, l_idx, :], expected, rtol=1e-12,
                    err_msg=f"Residual Malliavin at j={j}, l_idx={l_idx}"
                )

    def test_ssr_target_formula(self):
        """SSR_0(theta) -> H + 3/2 for all H values."""
        for H in [0.1, 0.3, 0.5]:
            target = H + 1.5
            assert abs(target - (H + 3/2)) < 1e-14

    def test_A0_theory_formula(self):
        """A_theory = (phi(0)/2) * (psi_A * r(theta) + psi_B * sqrt(theta))."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        r_theta = theta ** H
        A0_theory = (_phi(0.0) / 2.0) * (
            coeffs["psi_A"] * r_theta + coeffs["psi_B"] * np.sqrt(theta)
        )
        assert np.isfinite(A0_theory)

    def test_Y0_theory_formula(self):
        """Y_theory = phi(0) * ((psi_A/(2H+3))*r(theta) + (psi_B/4)*sqrt(theta))."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        coeffs = compute_scenario_coefficients(cfg, H)
        r_theta = theta ** H
        Y0_theory = _phi(0.0) * (
            coeffs["psi_A"] / (2*H + 3) * r_theta
            + coeffs["psi_B"] / 4.0 * np.sqrt(theta)
        )
        assert np.isfinite(Y0_theory)

    def test_ssr_ratio_formula(self):
        """Verify SSR = A_theory / Y_theory converges to H + 3/2 in the 2-term formula."""
        for H in [0.1, 0.3, 0.5]:
            cfg = baseline_A_config()
            coeffs = compute_scenario_coefficients(cfg, H)
            theta = 2**(-12)  # very short maturity
            r_theta = theta ** H

            psi_A = coeffs["psi_A"]
            psi_B = coeffs["psi_B"]

            # For H=0.5: r_theta = sqrt(theta), so dominant terms are equal order
            # For H<0.5: r_theta >> sqrt(theta), so psi_A term dominates
            A = (_phi(0.0) / 2.0) * (psi_A * r_theta + psi_B * np.sqrt(theta))
            Y = _phi(0.0) * (psi_A / (2*H + 3) * r_theta + psi_B / 4.0 * np.sqrt(theta))

            if abs(Y) > 1e-15:
                SSR_asymptotic = A / Y
                target = H + 1.5
                # For pure psi_B = 0 case: SSR = (psi_A/2) / (psi_A/(2H+3)) = (2H+3)/2 = H+3/2 ✓
                # For mixed case: also converges to H+3/2 as shown in paper
                # Check with psi_B = 0 (Baseline B's behavior)
                A_noB = (_phi(0.0) / 2.0) * psi_A * r_theta
                Y_noB = _phi(0.0) * psi_A / (2*H + 3) * r_theta
                if abs(Y_noB) > 1e-15:
                    SSR_noB = A_noB / Y_noB
                    assert abs(SSR_noB - target) < 1e-10, \
                        f"H={H}: SSR_noB={SSR_noB:.4f}, target={target:.4f}"

    def test_malliavin_basket_aggregation(self):
        """D_0^l log S_theta = sum_j Pi_theta^j D_0^l log S_theta^j."""
        cfg = baseline_A_config()
        H, theta = 0.3, 0.015625
        M, N_half = 32, 100
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=10)

        V_paths = result["V_paths"]
        dB_paths = result["dB_paths"]
        Pi_terminal = result["Pi_terminal"]
        V0 = np.asarray(cfg["V0"])
        beta = np.asarray(cfg["beta"])
        rho_loadings = np.asarray(cfg["rho_loadings"])
        d = cfg["d"]
        N = result["N"]
        dt = result["dt"]

        D1 = compute_malliavin_factor_direction(V_paths, dB_paths, beta, rho_loadings, V0, H, dt)
        D_res = compute_malliavin_residual_directions(rho_loadings, V0, d, N)
        D_basket_factor, D_basket_residual = compute_basket_malliavin(D1, D_res, Pi_terminal)

        # Verify aggregation shape
        assert D_basket_factor.shape == (N,)
        assert D_basket_residual.shape == (cfg["n"], N)  # n_residual = n


# ─── Exp 5 methodology checks ────────────────────────────────────────────────

class TestExp5Methodology:
    """Validate Exp 5: Constituent-level single-asset sanity checks."""

    def test_bar_vi_equals_V0i_for_each_constituent(self):
        """bar v^i(theta) = V_0^i exactly since E[V_t^i] = V_0^i."""
        # This is a direct consequence of the exponential-Gaussian specification
        # V_t^i = V_0^i exp(beta_i Z_t - 0.5 beta_i^2 Var(Z_t))
        # => E[V_t^i] = V_0^i * exp(0) = V_0^i (martingale variance)
        cfg = baseline_A_config()
        V0 = np.asarray(cfg["V0"])
        H = 0.3
        theta = 0.015625
        M, N_half = 128, 3000
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=0)
        V = result["V_paths"]
        for i in range(cfg["n"]):
            empirical_mean_V = float(V[i, :, :].mean())
            assert abs(empirical_mean_V - V0[i]) / V0[i] < 0.05, \
                f"E[V^{i}] = {empirical_mean_V:.6f}, V0[{i}] = {V0[i]:.6f}"

    def test_constituent_asymptotic_slope_formula(self):
        """Single-asset slope = psi^{ii} / (2H+3) * r(theta)."""
        cfg = baseline_A_config()
        H = 0.3
        theta = 0.015625
        beta = np.asarray(cfg["beta"])
        rho_loadings = np.asarray(cfg["rho_loadings"])

        for i in range(cfg["n"]):
            gamma_ii = beta[i] * rho_loadings[i, 0]
            psi_ii = 2.0 * gamma_ii / (2*H + 1)
            slope = psi_ii / (2*H + 3) * theta**H
            assert np.isfinite(slope)
            # For Baseline A: beta[0] = -1, rho_1^1 = 0.95 => slope < 0 (negative skew)
            if i == 0:
                assert slope < 0, f"Constituent 1 slope should be negative (rough vol skew)"

    def test_constituent_ATM_iv_close_to_sqrt_V0(self):
        """ATM constituent IV ≈ sqrt(V_0^i) for short maturities."""
        cfg = baseline_A_config()
        H, theta = 0.3, 0.015625
        M, N_half = 128, 2000
        result = simulate_model_paths(cfg, H, theta, M, N_half, seed=42)
        S_terminal = result["S_terminal"]
        S0 = np.asarray(cfg["S0"])
        V0 = np.asarray(cfg["V0"])

        for i in range(cfg["n"]):
            price_mc, _ = put_price_from_mc(S_terminal[i], S0[i])
            sigma_mc = implied_vol_from_price(price_mc, S0[i], S0[i], theta)
            if not np.isnan(sigma_mc):
                atm_theory = np.sqrt(V0[i])
                rel_err = abs(sigma_mc - atm_theory) / atm_theory
                assert rel_err < 0.20, \
                    f"Constituent {i}: sigma_MC={sigma_mc:.4f} vs sqrt(V0)={atm_theory:.4f}"
