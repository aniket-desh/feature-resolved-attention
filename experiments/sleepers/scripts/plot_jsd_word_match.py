"""1x2 plot: word-for-word match rate between steered and clean rollouts vs α.

Two stats per cell, both averaged over sae_seeds:
  • exact-match-rate: fraction of prompts where the full 16-token steered
    rollout equals the 16-token clean rollout (same decode RNG).
  • per-position match: fraction of (prompt × position) pairs where the
    steered token equals the clean token.

Same recipe-comparison and palette as plot_jsd_overlay.py: green/red reserved
for JSD-clean / JSD-poisoned in *that* plot, here we just need to distinguish
methods, so we use:
  blue solid + ●   single OV → OV  · exact-match
  blue dashed + ▲  conventional    · exact-match
(both stats on the same y-axis, range [0, 1])

A second axis or second panel could show the per-position match — for now
we render a single panel per SAE size with both methods' exact-match rate.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reduce_mean(values):
    if isinstance(values, list):
        return float(statistics.mean(values))
    return float(values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n_prompts", type=int, default=200,
                   help="number of eval prompts (used to convert n_exact → fraction)")
    p.add_argument("--metric", choices=["exact", "pos"], default="exact",
                   help="exact = fraction of prompts whose 16-token rollout matches "
                        "clean; pos = fraction of (prompt×position) tokens matching")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)

    OV_C   = "#1f77b4"
    CONV_C = "#1f77b4"   # same color, different linestyle (single-method-vs-method palette)

    panel_setup = [
        (axes[0], "(a)  4k SAE",  "ov_single_4k",  "conventional_4k"),
        (axes[1], "(b)  50k SAE", "ov_single_50k", "conventional_50k"),
    ]

    for ax, title, ov_key, conv_key in panel_setup:
        for key, name, linestyle, marker in [
            (ov_key,   "single OV → OV",  "-",  "o"),
            (conv_key, "conventional",    "--", "^"),
        ]:
            entry = cfg[key]
            per_alpha = entry["per_alpha"]
            if args.metric == "exact":
                ys = [reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                       for a in alphas]
                ylabel = f"fraction of {args.n_prompts} prompts whose 16-token rollout\nmatches clean word-for-word"
            else:
                ys = [reduce_mean(per_alpha[str(a)]["frac_pos_match_clean"]) for a in alphas]
                ylabel = "per-position match rate (steered token = clean token)"

            ax.plot(alphas, ys, color="#1f77b4", lw=2.3, marker=marker, markersize=7,
                    linestyle=linestyle, label=name)

        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2g}" for a in alphas], fontsize=9)
        ax.set_xlabel("steering coefficient α", fontsize=11)
        ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95)

    axes[0].set_ylabel(ylabel, fontsize=11)

    if args.title:
        suptitle = args.title
    else:
        kind = ("exact 16-token" if args.metric == "exact" else "per-position")
        suptitle = (
            f"Steered-vs-clean rollout {kind} match rate vs α  ·  mean over 3 SAE seeds  ·  "
            f"sampling RNG matched (decode_seed=0)"
        )
    fig.suptitle(suptitle, fontsize=11.5, y=1.00)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    print(f"\n{'sae':<8} {'method':<14} {'α':>5}  "
          f"{'mean exact-match':>16}  {'mean pos-match':>14}")
    for ax, title, ov_key, conv_key in panel_setup:
        for key, name in [(ov_key, "OV→OV"), (conv_key, "conventional")]:
            entry = cfg[key]
            per_alpha = entry["per_alpha"]
            for a in alphas:
                em = reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                pm = reduce_mean(per_alpha[str(a)]["frac_pos_match_clean"])
                print(f"{title.split()[1]:<8} {name:<14} {a:>5.2f}  "
                      f"{em:>16.4f}  {pm:>14.4f}")


if __name__ == "__main__":
    main()
