#!/usr/bin/env python3
"""
Paper-style headline plot for the Cadenza sleeper FRA-vs-conventional
comparison. Two panels side-by-side:

  - Left:  test ASR ↓  (lower = better suppression)
  - Right: test JSD vs clean ↓  (lower = less coherence cost)

Bars: FRA OV via hook_v, FRA QK→OV (from the 3×3 matrix), and three
conventional additive cells at L29 (ln1 / resid_mid / resid_post).
Colour scheme matches `scripts/plot_phase1_headline_per_domain.py`:

  - FRA QK→QK : green   #009E73
  - FRA QK→OV : blue    #0072B2
  - FRA OV→OV : orange  #D55E00
  - conventional : black  #000000

Inputs:
  - logs/cadenza_phase3/4way_metrics.json
  - logs/cadenza_phase2/step2_attribution_matrix.json (for QK→OV row)

Output:
  - experiments/sleepers/cadenza/figures/fra_vs_conventional.png  (+ .pdf)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLOR_BY_RECIPE = {
    "qk_to_qk":     "#009E73",
    "qk_to_ov":     "#0072B2",
    "ov_to_ov":     "#D55E00",
    "conventional": "#000000",
}


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           14,
        "axes.titlesize":      17,
        "axes.labelsize":      15,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.2,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "xtick.color":         "#222222",
        "ytick.color":         "#222222",
        "xtick.labelsize":     12,
        "ytick.labelsize":     13,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.12,
    })


def _best_row(per_alpha: dict, metric: str = "asr"):
    """Return (alpha, mean, std) over the α grid minimising mean ASR."""
    best_a = None; best_mean = 2.0; best_std = 0.0
    for a, vals in per_alpha.items():
        v = vals[metric]
        if isinstance(v, list):
            m = sum(v) / max(len(v), 1)
            s = (np.std(v, ddof=1) if len(v) >= 2 else 0.0)
        else:
            m = float(v); s = 0.0
        if m < best_mean:
            best_mean = m; best_std = s; best_a = a
    return best_a, best_mean, best_std


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fourway",
                   default="logs/cadenza_phase3/4way_metrics.json")
    p.add_argument("--matrix",
                   default="logs/cadenza_phase2/step2_attribution_matrix.json")
    p.add_argument("--out",
                   default="experiments/sleepers/cadenza/figures/fra_vs_conventional")
    args = p.parse_args()

    setup_style()

    fourway = json.loads(Path(args.fourway).read_text())
    matrix = json.loads(Path(args.matrix).read_text()) if Path(args.matrix).exists() else None

    # Collect rows in the order we want bars to appear (left → right):
    # FRA OV via hookv, FRA QK→OV (from matrix), conv ln1, conv mid, conv post.
    rows: list[dict] = []

    fw = fourway["configs"]
    # FRA OV via hookv
    if "fra_ov_via_hookv" in fw:
        c = fw["fra_ov_via_hookv"]
        a, asr_m, asr_s = _best_row(c["per_alpha"], "asr")
        _, jsd_m, jsd_s = _best_row({a: c["per_alpha"][a]}, "jsd_clean")
        rows.append({
            "label": r"FRA OV $\rightarrow$ OV",
            "color_key": "ov_to_ov",
            "alpha": float(a), "asr": asr_m, "asr_std": asr_s,
            "jsd": jsd_m, "jsd_std": jsd_s,
        })

    # FRA QK→OV from the 3×3 matrix (best cell that is qk-attribute / ov-intervene)
    if matrix:
        for row in matrix.get("matrix", []):
            if row["attribute"] == "qk" and row["intervene"] == "ov":
                rows.append({
                    "label": r"FRA QK $\rightarrow$ OV",
                    "color_key": "qk_to_ov",
                    "alpha": float(row["best_alpha"]),
                    "asr": float(row["best_asr"]), "asr_std": 0.0,
                    "jsd": float(row["best_jsd_clean"]), "jsd_std": 0.0,
                })
                break

    # Conventional additive at the three L29 hookpoints
    for cfg_name, label in (
        ("conv_additive_ln1",  "Conv. add. L29 ln1"),
        ("conv_additive_mid",  "Conv. add. L29 resid_mid"),
        ("conv_additive_post", "Conv. add. L29 resid_post"),
    ):
        if cfg_name not in fw:
            continue
        c = fw[cfg_name]
        a, asr_m, asr_s = _best_row(c["per_alpha"], "asr")
        _, jsd_m, jsd_s = _best_row({a: c["per_alpha"][a]}, "jsd_clean")
        rows.append({
            "label": label,
            "color_key": "conventional",
            "alpha": float(a), "asr": asr_m, "asr_std": asr_s,
            "jsd": jsd_m, "jsd_std": jsd_s,
        })

    assert rows, "no recipe data found"

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.6),
                                     gridspec_kw={"wspace": 0.32})
    x = np.arange(len(rows))
    colors = [COLOR_BY_RECIPE[r["color_key"]] for r in rows]
    labels = [r["label"] for r in rows]
    alphas = [r["alpha"] for r in rows]

    # ── ASR panel ──────────────────────────────────────────────────────
    asr_m = [r["asr"] for r in rows]
    asr_s = [r["asr_std"] for r in rows]
    bars = ax_l.bar(x, asr_m, yerr=asr_s, color=colors,
                    edgecolor="#222222", linewidth=1.0,
                    error_kw=dict(ecolor="#222222", capsize=6, capthick=1.4, lw=1.4),
                    zorder=3)
    for bar, m, a in zip(bars, asr_m, alphas):
        h = bar.get_height()
        ax_l.text(bar.get_x() + bar.get_width()/2,
                  max(h, 0.0) + 0.025, f"{m:.3f}",
                  ha="center", va="bottom",
                  fontsize=12, fontweight="bold", color="#0a0a0a", zorder=5)
        ax_l.text(bar.get_x() + bar.get_width()/2,
                  -0.06, rf"$\alpha^*$={a:+.1f}",
                  ha="center", va="top", fontsize=10, color="#666666")
    ax_l.set_ylabel("test ASR  (lower = better suppression)")
    ax_l.set_title("Attack-success rate (deployed)")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax_l.set_ylim(0.0, max(asr_m + [1.0]) * 1.15)
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.axhline(1.0, color="#999999", lw=0.8, ls=":", zorder=1)
    ax_l.text(len(rows) - 0.5, 1.02, "unsteered = 1.0",
              ha="right", va="bottom", fontsize=10, color="#888888")

    # ── JSD panel ──────────────────────────────────────────────────────
    jsd_m = [r["jsd"] for r in rows]
    jsd_s = [r["jsd_std"] for r in rows]
    bars = ax_r.bar(x, jsd_m, yerr=jsd_s, color=colors,
                    edgecolor="#222222", linewidth=1.0,
                    error_kw=dict(ecolor="#222222", capsize=6, capthick=1.4, lw=1.4),
                    zorder=3)
    for bar, m in zip(bars, jsd_m):
        h = bar.get_height()
        ax_r.text(bar.get_x() + bar.get_width()/2,
                  h + max(jsd_m) * 0.02, f"{m:.3f}",
                  ha="center", va="bottom",
                  fontsize=12, fontweight="bold", color="#0a0a0a", zorder=5)
    ax_r.set_ylabel("test JSD vs clean  (lower = less coherence cost)")
    ax_r.set_title("Coherence cost on clean prompts")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax_r.set_ylim(0.0, max(jsd_m) * 1.25)
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)

    fig.suptitle(
        "Cadenza Llama-3 8B sleeper @ L29 — FRA vs conventional additive steering",
        fontsize=16, y=1.02,
    )
    fig.text(0.5, -0.05,
             "Bars: best operating point per recipe over the α-sweep "
             "(min mean test ASR). "
             "Recipes coloured by paper convention. "
             "Conventional additive at resid_mid / resid_post matches FRA on "
             "ASR collapse; FRA wins specifically at ln1 (pre-attention).",
             ha="center", va="top", fontsize=10.5, color="#555555")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
