# Near-the-money Basket Implied Volatility Expansion

Reproducible Python implementation of the short-maturity near-the-money implied volatility
expansion and associated digital/density/SSR asymptotics for basket options under synthetic
**Gaussian Volterra-factor stochastic volatility models** (rough-volatility regime H ∈ {0.1, 0.3, 0.5}).

---

## Repository Layout

```
src/               Core library — all custom implementations
  coefficients.py  Analytic coefficients: ψ_A, ψ_B, v̄(θ), c_θ, baseline configs
  simulation.py    Volterra-factor path simulation, log-Euler asset prices
  blackscholes.py  BS put pricing, implied-vol inversion, ATM skew from digital
  malliavin.py     Pathwise Malliavin derivatives D_0^l log S_θ for SSR
  monte_carlo.py   Batch MC engine: smile, ATM skew, density, SSR helpers

exp/               Experiment scripts
  exp_1_smile.py        Near-the-money basket smile expansion (Exp 1)
  exp_2_atm_skew.py     ATM skew asymptotics + digital correction (Exp 2)
  exp_3_density.py      Short-maturity density expansion (Exp 3)
  exp_4_ssr.py          Skew-stickiness ratio (SSR) asymptotics (Exp 4)
  exp_5_constituent.py  Single-asset constituent sanity check (Exp 5)
  run_all_experiments.py Master orchestrator

data/              Synthetic data generation utilities
tests/             pytest suite (107 tests)
results/           Generated CSVs, PNGs, RESULTS.md
```

---

## Quick Start

```bash
# Install dependencies (already present in the container)
pip install numpy scipy matplotlib pandas joblib numba

# Fast smoke run — all 5 experiments, ~30 seconds
python exp/run_all_experiments.py --fast

# Full run — all H values, full maturity ladder, more paths (~10-15 min)
python exp/run_all_experiments.py

# Individual experiments
python exp/exp_1_smile.py [--fast]
python exp/exp_2_atm_skew.py [--fast]
python exp/exp_3_density.py [--fast]
python exp/exp_4_ssr.py [--fast]
python exp/exp_5_constituent.py [--fast]

# Unit tests
python -m pytest tests/ -v
```

---

## Experiments and Theory

### Exp 1 — Near-the-money Smile Expansion
Validates: **σ(√θ z, θ) ≈ √(v̄(θ)) + c_θ z**  
where **c_θ = (ψ_A/(2H+3)) θ^H + (ψ_B/4) √θ** decomposes the basket skew into
a volatility-fluctuation term (ψ_A) and a portfolio-weight-fluctuation term (ψ_B).

### Exp 2 — ATM Skew Asymptotics
Validates: **∂_k σ(0,θ) ≈ (ψ_A/(2H+3)) θ^{H-½} + ψ_B/4**  
Primary estimator: paper-faithful digital identity  
Q(S_θ < S₀) − Φ(√θ σ(0,θ)/2) = φ(d₊) · √θ · ∂_k σ(0,θ)  
Secondary: centered finite-difference cross-check.

### Exp 3 — Short-maturity Density Expansion
Validates: **q_θ(x) = φ(x + σ₀/2)[1 + c_θ H₃(x + σ₀/2)]** (H₃ = third Hermite polynomial)  
and the scalar digital correction Y₀^θ ≈ φ(0) c_θ.

### Exp 4 — SSR Universal Limit
Validates: **SSR₀(θ) = A₀^θ / Y₀^θ → H + 3/2**  
using the Malliavin-representation numerator  
**A₀^θ = E[1_{S_θ<S₀}(S_θ/S₀)(1 − (1/v₀) Σ_l Σ_i Π₀^i √(V₀^i) ρ_l^i D₀^l log S_θ)]**

### Exp 5 — Constituent-level Sanity Check
Validates: **σ^i(√θ z, θ) ≈ √(V₀^i) + [ψ^{ii}/(2H+3)] θ^H z**  
No basket-weight channel; isolates the volatility-fluctuation mechanism.

---

## Baselines

| Baseline | n | S₀        | Π₀        | ρ₁^i          | β_i         | psi_B |
|----------|---|-----------|-----------|---------------|-------------|-------|
| A        | 2 | (80, 20)  | (0.8,0.2) | (0.95, 0.10)  | (−1.0,−0.2) | >0    |
| B        | 2 | (50, 50)  | (0.5,0.5) | (0.70, 0.70)  | (−1.0,−1.0) | =0    |

Common: d=1, q=1, pure-power kernel K(u)=u^{H-½}, one Volterra factor.

---

## Key Results (fast mode, H=0.1)

| Experiment | Key finding | Status |
|------------|-------------|--------|
| 1 Smile    | Smile error decreases with θ; Baseline B (ψ_B=0) converges faster | ✅ |
| 2 ATM skew | Grows as θ^{H-½} for H<0.5; converges to finite limit for H=0.5 | ✅ |
| 3 Density  | Y₀^θ/c_θ → φ(0); digital Q matches theory | ✅ |
| 4 SSR      | SSR₀(θ) → 1.6 = H+3/2 for H=0.1, both baselines | ✅ |
| 5 Constituent | σ_MC(ATM) → √(V₀^i) = {0.10, 0.40} for A; both identical for B | ✅ |

All 107 unit tests pass. Full numerical results in [`results/RESULTS.md`](results/RESULTS.md).

---

## No External Data

All simulations are synthetic (risk-neutral measure Q). No market data API calls.
The `MASSIVE_API_KEY` environment variable is not required.

---

## Implementation Notes

- **Volterra convolution** discretised as Z_{t_m} = Σ_{j<m} ((m−j)dt)^{H-½} ΔW_j^1  
  using a pre-computed lower-triangular kernel matrix and numpy matmul.
- **Antithetic variates** halve variance; independent batches quantify MC error.
- **BS inversion** via `scipy.optimize.brentq` on [1e-8, 5.0].
- All formulas from the paper are implemented from scratch (no quant library covers  
  these paper-specific Gaussian-factor basket asymptotics).
- `numba` is used for JIT acceleration of the Volterra convolution inner loop.
