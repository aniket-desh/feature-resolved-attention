"""2x2 JSD plot with optional decode-seed-wise error bars.

Rows:
  top    — conventional steering (resid-mid additive, downstream feature 579)
  bottom — single-feature OV→OV steering (the selected pipeline winner, sae_seed=0)

Cols:
  left   — 4k SAE
  right  — 50k SAE

Each panel: JSD(steered, clean) and JSD(steered, poisoned) vs α (bits).

Two input shapes are supported:
  • single-seed: per-alpha entries `{"jsd_clean": float, "jsd_pois": float}`
  • multi-seed:  per-alpha entries `{"jsd_clean": [..], "jsd_pois": [..]}`
For multi-seed input, plots mean line + ±1σ shaded band.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reduce_seedwise(values):
    """Return (mean, std) given a float or list of floats."""
    if isinstance(values, list):
        if len(values) == 1:
            return float(values[0]), 0.0
        return float(statistics.mean(values)), float(statistics.stdev(values))
    return float(values), 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    cells = [
        [("conventional_4k",  "(a) conventional · resid-mid additive · 4k SAE"),
         ("conventional_50k", "(b) conventional · resid-mid additive · 50k SAE")],
        [("ov_single_4k",     "(c) single OV → OV · 4k SAE"),
         ("ov_single_50k",    "(d) single OV → OV · 50k SAE")],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, sharey=True)

    GREEN = "#2ca02c"
    RED   = "#d62728"
    n_seeds = len(data.get("decode_seeds", [0])) or 1

    for r, row in enumerate(cells):
        for c, (key, title) in enumerate(row):
            ax = axes[r, c]
            entry = cfg[key]
            # Handle both new (per_alpha) and old (results) shapes.
            per_alpha = entry.get("per_alpha") or entry.get("results")
            jc_mean, jc_std, jp_mean, jp_std = [], [], [], []
            for a in alphas:
                slot = per_alpha[str(a)]
                m, s = reduce_seedwise(slot["jsd_clean"])
                jc_mean.append(m); jc_std.append(s)
                m, s = reduce_seedwise(slot["jsd_pois"])
                jp_mean.append(m); jp_std.append(s)
            feat = entry["feature"]

            ax.plot(alphas, jc_mean, color=GREEN, lw=2.4, marker="o", markersize=7,
                    label="JSD(steered , clean)  — distance from clean (lower=better)")
            ax.fill_between(
                alphas, [m - s for m, s in zip(jc_mean, jc_std)],
                        [m + s for m, s in zip(jc_mean, jc_std)],
                color=GREEN, alpha=0.18, linewidth=0,
            )
            ax.plot(alphas, jp_mean, color=RED, lw=2.4, marker="s", markersize=7,
                    label="JSD(steered , poisoned)  — distance from sleeper (higher=better)")
            ax.fill_between(
                alphas, [m - s for m, s in zip(jp_mean, jp_std)],
                        [m + s for m, s in zip(jp_mean, jp_std)],
                color=RED, alpha=0.18, linewidth=0,
            )

            ax.axhline(1.0, color="#555", linestyle="--", lw=0.8, alpha=0.6)
            ax.text(alphas[-1], 1.0 - 0.02, "JSD upper bound (1 bit)", fontsize=9,
                    color="#555", ha="right", va="top")

            ax.set_title(f"{title}  (feature = {feat})", fontsize=11.5)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.04, 1.1)
            ax.set_xticks(alphas)
            ax.set_xticklabels([f"{a:.2g}" for a in alphas], fontsize=9)
            ax.legend(loc="center right", fontsize=9, frameon=True, framealpha=0.95)

    for c in (0, 1):
        axes[1, c].set_xlabel("steering coefficient α", fontsize=11)
    for r in (0, 1):
        axes[r, 0].set_ylabel("Jensen-Shannon divergence (bits)", fontsize=11)

    if args.title:
        suptitle = args.title
    else:
        suptitle = (
            f"JSD vs α — conventional (top) vs single OV → OV (bottom) · "
            f"shaded = ±1σ over {n_seeds} decode seeds\n"
            "left = 4k SAE · right = 50k SAE · sae_seed=0 · pipeline winners (f1114) · downstream f579"
        )
    fig.suptitle(suptitle, fontsize=11.5, y=1.00)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # Numeric table
    print(f"\n{'config':<22} {'α':>5}  {'jsd(s,clean) mean ± σ':>26}  {'jsd(s,pois) mean ± σ':>26}")
    for r, row in enumerate(cells):
        for c, (key, _) in enumerate(row):
            entry = cfg[key]
            per_alpha = entry.get("per_alpha") or entry.get("results")
            for a in alphas:
                slot = per_alpha[str(a)]
                jcm, jcs = reduce_seedwise(slot["jsd_clean"])
                jpm, jps = reduce_seedwise(slot["jsd_pois"])
                print(f"{key:<22} {a:>5.2f}  {jcm:>14.4f} ± {jcs:.4f}  {jpm:>14.4f} ± {jps:.4f}")


if __name__ == "__main__":
    main()
