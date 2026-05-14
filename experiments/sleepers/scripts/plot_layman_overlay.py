"""Layman analog of plot_jsd_overlay.py.

Replaces the two distribution-level JSD measurements with two intuitive
rollout-level counts:

  green = fraction of 200 prompts whose 16-token steered rollout matches the
          clean rollout word-for-word (analog of JSD(steered, clean) — high
          = better, steered preserves clean behavior)
  red   = ASR-16: fraction of prompts whose steered rollout contains the
          sleeper phrase regex (analog of JSD(steered, poisoned) — low =
          better, sleeper suppressed)

Style matches the EM headline figures (`figures/em_figures/phase1_*` on the
em-repl branch): Inter / Helvetica sans-serif, hidden top/right
spines, light gridlines behind data, rounded baseline annotation box. PNG
+ PDF output.

Layout: left = 4k SAE, right = 50k SAE. Means over sae_seeds; no error bars.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          15,
        "axes.titlesize":     18,
        "axes.labelsize":     16,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "#222222",
        "axes.labelcolor":    "#1a1a1a",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.labelsize":    13,
        "ytick.labelsize":    14,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     True,
        "legend.fontsize":    12,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def reduce_mean(values):
    if isinstance(values, list):
        return float(statistics.mean(values))
    return float(values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Path *without* extension; .png and .pdf are written.")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    setup_style()
    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=True)

    # sleeper-plot palette — red/green preserves clean-vs-sleeper semantics;
    # darker tones than the bright matplotlib defaults to match the EM-figure look
    GREEN = "#1a8a3f"   # clean-match rate (higher = better)
    RED   = "#c0322a"   # ASR (lower = better)

    panel_setup = [
        (axes[0], "ov_single_4k",  "conventional_4k",  "4k SAE"),
        (axes[1], "ov_single_50k", "conventional_50k", "50k SAE"),
    ]

    # baseline = α=0 reference (steered ≡ poisoned, no hook fires)
    baseline_asr = reduce_mean(cfg[panel_setup[0][1]]["per_alpha"]["0.0"]["asr"])
    baseline_match = (reduce_mean(cfg[panel_setup[0][1]]["per_alpha"]["0.0"]["n_exact_match_clean"])
                      / args.n_prompts)

    for ax, ov_key, conv_key, sae_label in panel_setup:
        for key, name, linestyle, marker in [
            (ov_key,   r"single OV$\rightarrow$OV", "-",  "o"),
            (conv_key, "conventional",              "--", "^"),
        ]:
            entry = cfg[key]
            per_alpha = entry["per_alpha"]
            match_rate = [reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                          for a in alphas]
            asr        = [reduce_mean(per_alpha[str(a)]["asr"]) for a in alphas]

            ax.plot(alphas, match_rate, color=GREEN, lw=2.6, marker=marker, markersize=8,
                    linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                    label=f"{name}  clean-match rate")
            ax.plot(alphas, asr, color=RED, lw=2.6, marker=marker, markersize=8,
                    linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                    label=f"{name}  sleeper rate (ASR)")

        ax.set_title(sae_label, loc="center", fontweight="bold", pad=14)
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2g}" for a in alphas])
        ax.set_xlabel(r"steering coefficient  $\alpha$")
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="#bbbbbb")

        # Unsteered baseline annotation (top-left of each panel)
        ax.text(0.02, 0.97,
                f"Unsteered baseline\nclean-match = {baseline_match*100:.1f}%, "
                f"sleeper rate = {baseline_asr*100:.1f}%",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=11, color="#444",
                bbox=dict(facecolor="white", edgecolor="#222", boxstyle="round,pad=0.4"))

    axes[0].set_ylabel("fraction of 200 deployment prompts")

    if args.title:
        fig.suptitle(args.title, fontsize=15, y=1.00)

    fig.tight_layout()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = str(out)
    if base.endswith(".png") or base.endswith(".pdf"):
        base = base.rsplit(".", 1)[0]
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png and {base}.pdf")

    # Numeric table for sanity
    print(f"\n{'sae':<5} {'method':<14} {'α':>5}  {'clean-match':>11}  {'ASR':>6}")
    for ax, ov_key, conv_key, sae_label in panel_setup:
        for key, name in [(ov_key, "OV→OV"), (conv_key, "conventional")]:
            per_alpha = cfg[key]["per_alpha"]
            for a in alphas:
                m = reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                r = reduce_mean(per_alpha[str(a)]["asr"])
                print(f"{sae_label:<5} {name:<14} {a:>5.2f}  {m:>11.4f}  {r:>6.3f}")


if __name__ == "__main__":
    main()
