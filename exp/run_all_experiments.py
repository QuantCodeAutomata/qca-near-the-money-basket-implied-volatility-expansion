"""
Master experiment runner.

Executes all five experiments in sequence and aggregates results into
results/RESULTS.md.

Usage:
    python exp/run_all_experiments.py [--fast]

Options:
    --fast    Use reduced parameter grids for quick validation (fewer batches/maturities).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1(fast: bool = False) -> dict:
    """Run Experiment 1: Near-the-money smile expansion."""
    print("\n" + "="*70)
    print("Experiment 1: Near-the-money basket IV smile expansion")
    print("="*70)
    t0 = time.time()

    from exp.exp_1_smile import main as exp1_main
    summary = exp1_main(fast=fast)

    elapsed = time.time() - t0
    print(f"  Exp 1 completed in {elapsed:.1f}s")
    return summary or {}


def run_exp2(fast: bool = False) -> dict:
    """Run Experiment 2: ATM skew asymptotics and digital correction."""
    print("\n" + "="*70)
    print("Experiment 2: ATM skew and digital correction")
    print("="*70)
    t0 = time.time()

    from exp.exp_2_atm_skew import main as exp2_main
    summary = exp2_main(fast=fast)

    elapsed = time.time() - t0
    print(f"  Exp 2 completed in {elapsed:.1f}s")
    return summary or {}


def run_exp3(fast: bool = False) -> dict:
    """Run Experiment 3: Short-maturity density expansion."""
    print("\n" + "="*70)
    print("Experiment 3: Short-maturity density expansion")
    print("="*70)
    t0 = time.time()

    from exp.exp_3_density import main as exp3_main
    summary = exp3_main(fast=fast)

    elapsed = time.time() - t0
    print(f"  Exp 3 completed in {elapsed:.1f}s")
    return summary or {}


def run_exp4(fast: bool = False) -> dict:
    """Run Experiment 4: SSR asymptotics."""
    print("\n" + "="*70)
    print("Experiment 4: Skew Stickiness Ratio (SSR) asymptotics")
    print("="*70)
    t0 = time.time()

    from exp.exp_4_ssr import main as exp4_main
    summary = exp4_main(fast=fast)

    elapsed = time.time() - t0
    print(f"  Exp 4 completed in {elapsed:.1f}s")
    return summary or {}


def run_exp5(fast: bool = False) -> dict:
    """Run Experiment 5: Single-asset constituent sanity check."""
    print("\n" + "="*70)
    print("Experiment 5: Constituent-level single-asset sanity check")
    print("="*70)
    t0 = time.time()

    from exp.exp_5_constituent import main as exp5_main
    summary = exp5_main(fast=fast)

    elapsed = time.time() - t0
    print(f"  Exp 5 completed in {elapsed:.1f}s")
    return summary or {}


def write_results_md(summaries: dict, fast: bool) -> None:
    """Write consolidated RESULTS.md."""
    lines = [
        "# Experiment Results",
        "",
        f"*Mode: {'fast (reduced parameters)' if fast else 'full'}*",
        "",
        "## Summary",
        "",
        "| Exp | Title | Status |",
        "|-----|-------|--------|",
    ]

    titles = {
        "exp_1": "Near-the-money basket IV smile expansion",
        "exp_2": "ATM skew asymptotics and digital correction",
        "exp_3": "Short-maturity density expansion",
        "exp_4": "SSR asymptotics and universal limit",
        "exp_5": "Constituent-level single-asset sanity check",
    }

    for key, title in titles.items():
        status = summaries.get(key, {}).get("status", "not run")
        emoji = "✅" if status == "passed" else ("❌" if status == "failed" else "⚠️")
        lines.append(f"| {key} | {title} | {emoji} {status} |")

    lines += ["", "## Theoretical Coefficients", ""]

    for key in titles:
        s = summaries.get(key, {})
        if "coefficients" in s:
            lines.append(f"### {key}: {titles[key]}")
            lines.append("")
            lines.append("| Baseline | H | psi_A | psi_B | v(0) | SSR target |")
            lines.append("|----------|---|-------|-------|------|-----------|")
            for row in s["coefficients"]:
                lines.append(
                    f"| {row.get('baseline','?')} | {row.get('H','?')} "
                    f"| {row.get('psi_A','?'):.4f} | {row.get('psi_B','?'):.4f} "
                    f"| {row.get('v0','?'):.6f} | {row.get('ssr_target','?')} |"
                )
            lines.append("")

    lines += ["## Plots", "", "All plots are saved to the `results/` directory:"]

    import os
    plot_files = sorted(Path("results").glob("*.png"))
    for f in plot_files:
        lines.append(f"- `{f.name}`")

    # Write
    md_path = RESULTS_DIR / "RESULTS.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults summary written to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all basket IV experiments")
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced parameters for quick validation")
    args = parser.parse_args()

    t_total = time.time()
    summaries: dict[str, dict] = {}

    experiments = [
        ("exp_1", run_exp1),
        ("exp_2", run_exp2),
        ("exp_3", run_exp3),
        ("exp_4", run_exp4),
        ("exp_5", run_exp5),
    ]

    for key, runner in experiments:
        try:
            summaries[key] = runner(fast=args.fast)
        except Exception as exc:
            print(f"\n[ERROR] {key} failed: {exc}")
            import traceback
            traceback.print_exc()
            summaries[key] = {"status": "failed", "error": str(exc)}

    # Save combined JSON
    json_path = RESULTS_DIR / "all_results.json"
    try:
        with open(json_path, "w") as f:
            json.dump(
                {k: {kk: str(vv) for kk, vv in v.items()} for k, v in summaries.items()},
                f, indent=2,
            )
    except Exception:
        pass

    write_results_md(summaries, fast=args.fast)

    total_elapsed = time.time() - t_total
    print(f"\nAll experiments completed in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
