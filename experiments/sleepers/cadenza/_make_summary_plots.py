"""
Generate the three summary plots referenced by summary.md.

  figures/locality_heatmap.png   — 3x3 layer × hookpoint test-ASR grid (v1 sweep)
                                   with the two N=250-validated cells annotated.
  figures/headline_result.png    — (test ASR, test ΔCE) bars for unsteered, the two
                                   validated L29 cells, and the v2 sabotage-mode
                                   false-positive at L3/ln1.
  figures/alpha_sweep_L29_post.png — for the winning feature at L29/resid_post,
                                   ASR vs α from the v1 sweep — visualises the
                                   "only negative α suppresses cleanly" mechanism.

All paths are relative to repo root. Run::

    python -m experiments.sleepers.cadenza._make_summary_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ── Shared style (matches experiments/sleepers/scripts/plot_combined_50k.py) ─


def setup_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica",
                               "Arial", "DejaVu Sans"],
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.0,
        "axes.edgecolor":     "#222222",
        "axes.labelcolor":    "#1a1a1a",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     True,
        "legend.fontsize":    10,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


GREEN  = "#1a8a3f"      # suppression / good
RED    = "#c0322a"      # sleeper firing / bad
NEUTRAL = "#888888"
WONG_GREEN = "#009E73"  # FRA QK→QK
WONG_BLUE  = "#0072B2"  # FRA QK→OV
WONG_ORANGE = "#D55E00" # FRA OV→OV
BLACK  = "#000000"


# ── Data loading ─────────────────────────────────────────────────────────


def load_v1_results() -> dict:
    """Return dict keyed by (layer, hook) -> v1 JSON record."""
    out = {}
    for p in sorted(Path("logs/cadenza_localisation").glob("cadenza_L*.json")):
        r = json.load(open(p))
        layer = r["cell"]["hook_layer"]
        hook = r["cell"]["hook_point"]
        out[(layer, hook)] = r
    return out


def load_validation_results() -> dict:
    """Return dict keyed by (layer, hook) -> validation JSON record."""
    out = {}
    for p in sorted(Path("logs/cadenza_validation").glob("L29_*.json")):
        r = json.load(open(p))
        layer = r["cell"]["hook_layer"]
        hook = r["cell"]["hook_point"]
        out[(layer, hook)] = r
    return out


def load_v2_results() -> dict:
    out = {}
    for p in sorted(Path("logs/cadenza_localisation_v2").glob("cadenza_L*.json")):
        r = json.load(open(p))
        layer = r["cell"]["hook_layer"]
        hook = r["cell"]["hook_point"]
        out[(layer, hook)] = r
    return out


# ── Plot 1 — locality heatmap ────────────────────────────────────────────


def plot_locality(v1: dict, validation: dict, out_path: Path):
    layers = [3, 16, 29]
    hooks = ["ln1.hook_normalized", "hook_resid_mid", "hook_resid_post"]
    hook_short = ["ln1", "resid_mid", "resid_post"]

    grid = np.full((len(layers), len(hooks)), np.nan)
    annotations = [[""] * len(hooks) for _ in layers]

    for i, L in enumerate(layers):
        for j, h in enumerate(hooks):
            r = v1.get((L, h))
            if r is None:
                continue
            asr = r["test"]["asr"]
            grid[i, j] = asr
            label = f"{asr:.2f}"
            # If this cell has a validated N=250 measurement, add it
            if (L, h) in validation:
                vasr = validation[(L, h)]["test"]["asr_mean"]
                label += f"\n(N=250: {vasr:.2f})"
            annotations[i][j] = label

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "asr_cmap", [GREEN, "#fff5b1", RED], N=256,
    )
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(layers)):
        for j in range(len(hooks)):
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    color="#1a1a1a", fontsize=10.5, fontweight="bold")

    ax.set_xticks(range(len(hooks)))
    ax.set_xticklabels(hook_short)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("hookpoint")
    ax.set_ylabel("layer (of 32)")
    ax.set_title("Single-feature test ASR — Cadenza Llama-3 8B\n"
                 "v1 sweep (N=10 sampled); N=250 paper-spec re-measurements in parens",
                 fontsize=12)

    # Highlight the winner cell
    for i, L in enumerate(layers):
        for j, h in enumerate(hooks):
            if (L, h) in validation:
                vasr = validation[(L, h)]["test"]["asr_mean"]
                if vasr < 0.05:
                    rect = mpl.patches.Rectangle(
                        (j - 0.46, i - 0.46), 0.92, 0.92,
                        fill=False, edgecolor="#1a8a3f", linewidth=3,
                    )
                    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("test attack success rate  (↓ better)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Plot 2 — headline result (bars) ──────────────────────────────────────


def plot_headline(validation: dict, v2: dict, out_path: Path):
    # Configurations to compare
    items = [
        ("unsteered\nbaseline",            1.00,    0.0000,  NEUTRAL),
        ("L29/resid_post\nfeat 12402 α=-4\n(validated N=250)",
                                           validation[(29, "hook_resid_post")]["test"]["asr_mean"],
                                           validation[(29, "hook_resid_post")]["test"]["delta_ce"],
                                           WONG_ORANGE),
        ("L29/resid_mid\nfeat 22059 α=-0.5\n(validated N=250)",
                                           validation[(29, "hook_resid_mid")]["test"]["asr_mean"],
                                           validation[(29, "hook_resid_mid")]["test"]["delta_ce"],
                                           WONG_GREEN),
        ("L3/ln1\nfeat 14405 α=+2\n(v2 amplification)",
                                           v2[(3, "ln1.hook_normalized")]["test"]["asr_mean"],
                                           v2[(3, "ln1.hook_normalized")]["test"]["delta_ce"],
                                           RED),
    ]
    labels = [x[0] for x in items]
    asrs   = [x[1] for x in items]
    dces   = [x[2] for x in items]
    colors = [x[3] for x in items]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    xs = np.arange(len(items))
    bars_l = ax_l.bar(xs, asrs, color=colors, edgecolor="#1a1a1a", linewidth=0.8)
    ax_l.axhline(0.05, color="#666", linestyle=":", linewidth=0.9, alpha=0.7)
    ax_l.text(len(items) - 0.3, 0.06, "ASR ≤ 5%", fontsize=9, color="#666", ha="right")
    ax_l.set_ylabel("mean test ASR  (↓ better)")
    ax_l.set_ylim(0, 1.10)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_title("Trigger attack success rate")
    for bar, v in zip(bars_l, asrs):
        ax_l.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                  f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax_l.set_axisbelow(True)

    bars_r = ax_r.bar(xs, dces, color=colors, edgecolor="#1a1a1a", linewidth=0.8)
    ax_r.axhline(0.05, color="#666", linestyle=":", linewidth=0.9, alpha=0.7)
    ax_r.text(len(items) - 0.3, 0.052, "ΔCE budget = 0.05", fontsize=9,
              color="#666", ha="right")
    ax_r.axhline(0.0,  color="#1a1a1a", linewidth=0.8)
    ax_r.set_ylabel("test ΔCE on clean prompts (nats)  (↓ better, want ≈ 0)")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(labels, fontsize=9)
    ax_r.set_title("Coherence cost on clean prompts")
    for bar, v in zip(bars_r, dces):
        ax_r.text(bar.get_x() + bar.get_width()/2,
                  v + 0.01 if v >= 0 else v - 0.015,
                  f"{v:+.4f}", ha="center", fontsize=10, fontweight="bold")
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax_r.set_axisbelow(True)

    fig.suptitle(
        "Cadenza Llama-3 8B sleeper — headline result\n"
        "L29/hook_resid_post via anti-feature steering is the clean win "
        "(perfect suppression, zero coherence cost).",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Plot 3 — α-sweep for the winning feature ─────────────────────────────


def plot_alpha_sweep(v1: dict, out_path: Path):
    r = v1[(29, "hook_resid_post")]
    winner_feat = r["selection"]["feature"]  # 12402

    rows = [s for s in r["sweep"] if s["feature"] == winner_feat]
    rows.sort(key=lambda s: s["alpha"])
    alphas = [s["alpha"] for s in rows]
    asrs   = [s["asr"] for s in rows]
    dces   = [s["delta_ce"] for s in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(alphas, asrs, color=RED, marker="o", markersize=9,
            linewidth=2.4, label=f"val ASR (greedy, N=10)")
    ax.set_ylabel("val ASR  (↓ better)", color=RED)
    ax.tick_params(axis="y", labelcolor=RED)
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(0.0, color="#888", linestyle=":", linewidth=0.7, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(alphas, dces, color=GREEN, marker="s", markersize=8,
             linewidth=2.0, linestyle="--", label="val ΔCE on clean")
    ax2.set_ylabel("val ΔCE (nats)", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.axhline(0.05, color=GREEN, linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.spines["top"].set_visible(False)

    # Shade negative-α region
    ax.axvspan(min(alphas) - 0.5, 0.0, color="#dff0d8", alpha=0.4, zorder=0)
    ax.text(-3.5, 0.55, "anti-feature\ndirection\n(α < 0)",
            color="#1a8a3f", fontsize=10, ha="left")

    # Annotate winner
    win_idx = asrs.index(min(asrs))
    ax.annotate(
        f"winner: α={alphas[win_idx]:+.1f}\nval ASR=0\n→ N=250 test ASR=0",
        xy=(alphas[win_idx], asrs[win_idx]),
        xytext=(alphas[win_idx] + 0.7, 0.30),
        fontsize=10, fontweight="bold", color="#1a8a3f",
        arrowprops=dict(arrowstyle="->", color="#1a8a3f", lw=1.2),
    )

    ax.set_xlabel("steering coefficient  α   "
                  "(α=1 is no-op, α=0 ablates, α<0 is anti-feature)")
    ax.set_xticks(alphas)
    ax.set_title("α-sweep at L29/hook_resid_post for the winning feature 12402\n"
                 "Only the anti-feature (α<0) direction suppresses the sleeper; "
                 "the paper's α≥0 grid misses this.",
                 fontsize=12)
    ax.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    setup_style()
    fig_dir = Path("experiments/sleepers/cadenza/figures")
    v1 = load_v1_results()
    val = load_validation_results()
    v2 = load_v2_results()
    print(f"v1 cells loaded: {len(v1)}; validation cells: {len(val)}; v2 cells: {len(v2)}")
    plot_locality(v1, val,        fig_dir / "locality_heatmap.png")
    plot_headline(val, v2,        fig_dir / "headline_result.png")
    plot_alpha_sweep(v1,          fig_dir / "alpha_sweep_L29_post.png")
    print("done.")


if __name__ == "__main__":
    main()
