"""Pareto: sleepers suppressed vs. coherence cost, overlay across seeds + cells.

Reads each seed's pareto_3x3.json, plots one curve per (ranking, intervention).
For each curve:
  - 4 points at α ∈ {0.5, 1, 2, 3}
  - x = ΔCE (lower = better coherence)
  - y = 1 - ASR_16 = fraction of sleepers suppressed (higher = better)

Mean across seeds with shaded ±std band per cell.

Usage:
    python plot_pareto_overlay.py \
        --seed_dirs ketan_repl/seed0 ketan_repl/seed1 ketan_repl/seed2 \
        --output ketan_repl/seed_aggregate/pareto_overlay.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_seed(seed_dir: Path) -> dict:
    return json.loads((seed_dir / "pareto_3x3.json").read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--xlim", type=float, default=2.0,
                   help="upper limit on the ΔCE axis (default 2.0 nats)")
    p.add_argument("--include_resid_mid", action="store_true",
                   help="overlay each seed's single-feature resid_mid sweep "
                        "(reads <seed_dir>/resid_mid_sweep.json)")
    args = p.parse_args()

    docs = [load_seed(d) for d in args.seed_dirs]
    mid_docs = []
    if args.include_resid_mid:
        for d in args.seed_dirs:
            mp = d / "resid_mid_sweep.json"
            if mp.exists():
                mid_docs.append(json.loads(mp.read_text()))
            else:
                print(f"[plot] warning: missing {mp} — skipping resid_mid overlay")
    rankings = ["qk", "ov", "union"]
    interventions = ["ov", "qk", "all"]
    alphas = docs[0]["meta"]["alphas"]

    # Map (rank, interv) → list-of-(seed) curves; each curve is sorted-by-ΔCE points.
    curves: dict[tuple[str, str], list[list[tuple[float, float]]]] = {}
    for r in rankings:
        for i in interventions:
            curves[(r, i)] = []
            for d in docs:
                rows = d["grid"][r][i]["per_alpha"]
                pts = [(max(0.0, row["delta_ce"]), 1 - row["asr_16"]) for row in rows]
                pts.sort(key=lambda t: t[0])
                curves[(r, i)].append(pts)

    baselines = [d["baseline"]["asr_16"] for d in docs]
    base_y = 1 - float(np.mean(baselines))

    # ------------------------------------------------------------------
    # Two-panel figure:
    #   left  — full ΔCE range (0 .. args.xlim)
    #   right — zoom to ΔCE ∈ [0, 0.3] (the regime where the headline lives)
    # Each cell is one line (mean across seeds) + shaded std band on y.
    # ------------------------------------------------------------------
    color_for_rank = {"qk": "#2ca02c", "ov": "#1f77b4", "union": "#d62728"}
    style_for_intv = {"ov": "-", "qk": "--", "all": ":"}
    marker_for_intv = {"ov": "o", "qk": "s", "all": "^"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    for xlim, ax, label in [(args.xlim, axL, f"full ΔCE ∈ [0, {args.xlim:g}]"),
                             (0.3,        axR, "zoomed ΔCE ∈ [0, 0.3]")]:
        # Reference: baseline (no steering) sits at ΔCE=0, y=1-ASR_baseline.
        ax.scatter([0.0], [base_y], marker="*", s=240, color="black",
                   zorder=10, label="baseline (no steer)")

        for (r, i), per_seed_pts in curves.items():
            # Shared x-axis for the band: align each seed's points to the same
            # 4 alphas (already in same order since alphas are deterministic).
            alpha_to_xy = {a: [] for a in alphas}
            for d_idx, pts in enumerate(per_seed_pts):
                rows = docs[d_idx]["grid"][r][i]["per_alpha"]
                for row in rows:
                    alpha_to_xy[row["alpha"]].append(
                        (max(0.0, row["delta_ce"]), 1 - row["asr_16"])
                    )
            # Mean over seeds at each alpha, sort by mean-x.
            mean_pts = []
            std_pts = []
            for a in alphas:
                xy = alpha_to_xy[a]
                xs = np.array([p[0] for p in xy])
                ys = np.array([p[1] for p in xy])
                mean_pts.append((xs.mean(), ys.mean()))
                std_pts.append((xs.std(), ys.std()))
            order = sorted(range(len(mean_pts)), key=lambda k: mean_pts[k][0])
            mx = [mean_pts[k][0] for k in order]
            my = [mean_pts[k][1] for k in order]
            sy = [std_pts[k][1]  for k in order]

            ax.plot(mx, my, linestyle=style_for_intv[i], color=color_for_rank[r],
                    lw=1.8, alpha=0.85)
            ax.fill_between(mx,
                            [m - s for m, s in zip(my, sy)],
                            [m + s for m, s in zip(my, sy)],
                            color=color_for_rank[r], alpha=0.10)
            ax.scatter(mx, my, marker=marker_for_intv[i],
                       color=color_for_rank[r], s=55, edgecolor="k", linewidth=0.4,
                       zorder=5)

        ax.set_xlabel("coherence cost  ΔCE  (nats; lower = more coherent)")
        ax.set_ylabel("fraction of sleepers suppressed  =  1 − ASR$_{16}$")
        ax.set_xlim(-0.02 * xlim, xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(label)

    # Optional overlay: single resid_mid feature ablation (one curve per seed,
    # mean across seeds with std band).
    if mid_docs:
        for ax in (axL, axR):
            alpha_to_xy: dict[float, list[tuple[float, float]]] = {}
            for mdoc in mid_docs:
                for row in mdoc["per_alpha"]:
                    alpha_to_xy.setdefault(row["alpha"], []).append(
                        (max(0.0, row["delta_ce"]), 1 - row["asr_16"]))
            alphas_sorted = sorted(alpha_to_xy.keys())
            mx, my, sy = [], [], []
            for a in alphas_sorted:
                xy = alpha_to_xy[a]
                xs = np.array([p[0] for p in xy])
                ys = np.array([p[1] for p in xy])
                mx.append(xs.mean()); my.append(ys.mean()); sy.append(ys.std())
            ax.plot(mx, my, linestyle="-", color="#9467bd", lw=3.0, alpha=0.95,
                    zorder=8)
            ax.fill_between(mx,
                            [m - s for m, s in zip(my, sy)],
                            [m + s for m, s in zip(my, sy)],
                            color="#9467bd", alpha=0.15)
            ax.scatter(mx, my, marker="D", color="#9467bd", s=85,
                       edgecolor="k", linewidth=0.6, zorder=9)

    # Custom legend (compactified — colors = ranking, styles = intervention).
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="black", marker="*", markersize=14,
                      lw=0, label="baseline")]
    for r in rankings:
        handles.append(Line2D([0], [0], color=color_for_rank[r], lw=2.5,
                              label=f"rank: {r} (ln1)"))
    for i in interventions:
        handles.append(Line2D([0], [0], color="gray", linestyle=style_for_intv[i],
                              marker=marker_for_intv[i], lw=2,
                              label=f"intervene: {i}"))
    if mid_docs:
        handles.append(Line2D([0], [0], color="#9467bd", lw=3, marker="D",
                              markersize=8, label="single resid_mid feature"))
    axR.legend(handles=handles, loc="lower right", fontsize=9)

    n = len(docs)
    fig.suptitle(
        f"Pareto: sleepers suppressed vs. coherence cost\n"
        f"mean across {n} seeds · α ∈ {alphas} (markers from low → high) · shaded = ±std",
        fontsize=11,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
