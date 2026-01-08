#!/usr/bin/env python3
"""Compare results from all synthetic data experiments."""

import json
from pathlib import Path
from tabulate import tabulate

def load_results(experiments_dir: Path) -> dict:
    """Load all results.json files from experiment directories."""
    results = {}
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results[exp_dir.name] = json.load(f)
    return results

def compare_experiments(experiments_dir: Path = Path("experiments")):
    """Generate comparison table."""
    results = load_results(experiments_dir)

    if not results:
        print("No results found!")
        return

    # Define baseline for comparison
    baseline_map = results.get("baseline_no_aug", {}).get("mAP50", 0)
    trad_aug_map = results.get("traditional_augmentation", {}).get("mAP50", 0)

    # Build comparison table
    headers = ["Experiment", "mAP50", "mAP50-95", "Precision", "Recall", "vs Baseline", "vs Trad Aug"]
    rows = []

    for name, data in sorted(results.items()):
        map50 = data.get("mAP50", 0)
        map50_95 = data.get("mAP50-95", 0)
        precision = data.get("precision", 0)
        recall = data.get("recall", 0)

        vs_baseline = ((map50 - baseline_map) / baseline_map * 100) if baseline_map else 0
        vs_trad = ((map50 - trad_aug_map) / trad_aug_map * 100) if trad_aug_map else 0

        rows.append([
            name,
            f"{map50:.4f}",
            f"{map50_95:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{vs_baseline:+.1f}%",
            f"{vs_trad:+.1f}%"
        ])

    print("\n" + "=" * 80)
    print("SYNTHETIC DATA COMPARISON RESULTS")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline (no aug):    mAP50 = {baseline_map:.4f}")
    print(f"Traditional aug:      mAP50 = {trad_aug_map:.4f}")

    # Find best synthetic method
    synthetic_methods = {k: v for k, v in results.items()
                        if k not in ["baseline_no_aug", "traditional_augmentation"]}

    if synthetic_methods:
        best_method = max(synthetic_methods.items(), key=lambda x: x[1].get("mAP50", 0))
        print(f"\nBest synthetic method: {best_method[0]} (mAP50 = {best_method[1].get('mAP50', 0):.4f})")

        if best_method[1].get("mAP50", 0) > trad_aug_map:
            print("✅ Synthetic data IMPROVED over traditional augmentation!")
        else:
            print("❌ Synthetic data did NOT improve over traditional augmentation")

    return results

if __name__ == "__main__":
    compare_experiments()
