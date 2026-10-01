# Project: Gaussian Volterra-Factor Basket Model — Agent Notes

## Repository Layout

```
src/
  __init__.py
  coefficients.py   — psi_A, psi_B, bar_v, c_theta, baseline configs
  simulation.py     — simulate_model_paths (Volterra kernel, log-Euler)
  blackscholes.py   — BS put price, implied_vol_from_price, atm_implied_vol,
                       digital_probability, atm_skew_from_digital
  malliavin.py      — Malliavin derivative helpers for SSR experiment
  monte_carlo.py    — run_smile_experiment, run_atm_skew_experiment,
                       run_ssr_experiment, run_density_experiment
exp/
  exp_1_smile.py    — smile slice validation (sigma_MC vs expansion)
  exp_2_atm_skew.py — ATM skew / digital correction validation
results/            — CSV and PNG outputs (gitignored)
data/               — synthetic data generator
tests/              — test suite
```

## Key src API

### `src.coefficients`
- `baseline_A_config()` / `baseline_B_config()` → config dict (n=2, d=1)
- `compute_scenario_coefficients(config, H)` → dict with psi_A, psi_B, v0,
  rho, gamma, psi_ki, d_i, Pi0, etc.
- `compute_c_theta(psi_A, psi_B, H, theta)` → c_theta scalar/array
- `compute_bar_v(Pi0, V0, beta, rho, H, theta)` → scalar

### `src.monte_carlo`
- `run_smile_experiment(config, H, theta, z_grid, M, N_half, n_batches, base_seed)`
  Returns DataFrame: z, sigma_MC, sigma_full, sigma_A_only, sigma_B_only,
  abs_error, scaled_residual, se_MC, ci_lo, ci_hi
- `run_atm_skew_experiment(config, H, theta, M, N_half, n_batches, base_seed)`
  Returns dict: theta, H, sigma_ATM, sigma_ATM_se, digital, digital_se,
  Y0, Y0_se, Y0_theory, skew_digital, skew_fd, skew_fd_se, skew_theory,
  r_theta, psi_A, psi_B, c_theta
- Module constants: `MATURITY_LADDER` (2^{-6}…2^{-12}), `Z_GRID`

## Experiment Files (exp/)

All exp files must prepend the project root to sys.path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```
Use `matplotlib.use('Agg')` before importing pyplot for headless rendering.

### Fast mode pattern (applied to ALL exp files)
Every `run_*_for_scenario(config_name, H, M, N_half, n_batches, ..., fast=False)`:
```python
if fast:
    N_half = min(N_half, 500)
    n_batches = min(n_batches, 5)
maturities = MATURITY_LADDER[::2] if fast else MATURITY_LADDER
```
Every `main(fast: bool = False) -> dict`:
```python
H_list = [0.1] if fast else H_VALUES
```
All five `main()` functions return a dict with at least `{"status": ..., "n_scenarios": ...}`.
`run_all_experiments.py` calls each `main(fast=args.fast)`.

### exp_1_smile.py
- `MATURITY_LADDER = [2**-m for m in range(6, 13)]`
- `Z_GRID = [-2, -1, -0.5, 0, 0.5, 1, 2]`
- `run_smile_for_scenario(config_name, H, M=256, N_half=2000, n_batches=20, base_seed=0, fast=False)`
- `plot_smile_slices(df, config_name, H, save_path)` — 2-panel: smile + residual
- `main(fast)` — loops A/B × H_list; saves smile_baseline{A,B}_H{H}.csv/.png

### exp_2_atm_skew.py
- `run_atm_skew_for_scenario(config_name, H, M=256, N_half=2000, n_batches=20, base_seed=100, fast=False)`
  Returns DataFrame columns: theta, H, sigma_ATM, digital, Y0, Y0_theory,
  skew_digital, skew_fd, skew_theory, psi_A, psi_B, r_theta, sqrt_theta
- `plot_atm_skew(df, config_name, H, save_path)` — 2-panel:
  Left: θ^{1/2-H}·skew vs θ (H<0.5) or raw skew vs θ (H=0.5)
  Right: Y0/(r+√θ) vs θ with theory asymptote
- `main(fast)` — saves atm_skew_baseline{A,B}_H{H}.csv/.png + atm_skew_summary.csv

### exp_3_density.py
- `run_density_for_scenario(config_name, H, M=256, N_half=2000, n_batches=20, base_seed=200, fast=False)`
  Returns DataFrame columns: theta, H, sigma_0_theta, c_theta, Y0, Y0_theory,
  digital_mean, psi_A, psi_B
- `main(fast)` — saves density_baseline{A,B}_H{H}.csv/.png

### exp_4_ssr.py
- `run_ssr_for_scenario(config_name, H, M=256, N_half=2000, n_batches=20, base_seed=300, fast=False)`
  Returns DataFrame columns: theta, H, A0, A0_se, Y0, Y0_se, SSR, SSR_target,
  A0_theory, Y0_theory, psi_A, psi_B, r_theta
  pass/fail: SSR within 2·SE of target at smallest theta → PASS
- Requires src.malliavin for D_0^1 log S_theta^j computation
- `main(fast)` — saves ssr_baseline{A,B}_H{H}.csv/.png

### exp_5_constituent.py
- `run_constituent_for_scenario(config_name, H, M=256, N_half=2000, n_batches=20, base_seed=400, fast=False)`
  Returns DataFrame columns: config, H, theta, constituent, z, S0_i, K_i,
  sigma_MC, sigma_asym, residual, scaled_residual, psi_ii, r_theta
- `main(fast)` — saves constituent_baseline{A,B}_H{H}.csv/.png

## src.malliavin
- `compute_malliavin_derivative(V_paths, dB1_paths, K_weights, beta, rho1, V0, dt)`:
  Computes D_0^1 log S_theta^j for the single factor direction using the formula
  D_0^1 log S_theta^j = rho_1^j sqrt(V_0^j) + 0.5·∫√V^j β_j K(s) dB_s^j − 0.5·∫V^j β_j K(s) ds
  Stochastic integral approximated by left-point Riemann sum on simulation grid.
- Residual directions l>d: D_0^l log S_theta^j = rho_l^j sqrt(V_0^j) (constant, no stochastic part)

## Mathematical Background (short-maturity regime)

- Volterra factor: Z_t = ∫₀ᵗ (t−s)^{H−1/2} dW_s¹
- Var(Z_t) = t^{2H}/(2H)  [works for all H ∈ (0,0.5], including H=0.5 → t]
- r(θ) = θ^H;  for H<0.5, r(θ) dominates √θ as θ→0
- c_theta = (psi_A/(2H+3))·r(θ) + (psi_B/4)·√θ
- σ(√θ z, θ) ≈ √(v̄(θ)) + c_theta·z  (smile expansion, Exp 1)
- ATM skew = (psi_A/(2H+3))·θ^{H-1/2} + psi_B/4  (Exp 2)
- Y0 = Q(S_θ < S_0) − Φ(√θ·σ_ATM/2) ≈ φ(0)·c_theta  (Exps 2, 3)
- SSR₀(θ) = A₀^θ / Y₀^θ → H+3/2  (Exp 4, universal limit)
- σ^i(√θ z, θ) ≈ √(V₀^i) + [psi^{ii}/(2H+3)]·θ^H·z  (Exp 5, single asset)

## Known Fast-mode Limitations
- With n_batches=5 the FD skew estimator in exp_2 can produce RuntimeWarning
  about ddof<=0 when nanstd is applied to a single-value array — harmless.
- SSR convergence to H+3/2 needs more paths for tight CIs; fast mode shows the
  trend but the CI often still spans the target rather than tightly bracketing it.
- Smile errors in Baseline A are large in fast mode due to MC noise — they shrink
  with more paths and confirm asymptotic trend even in fast mode.
