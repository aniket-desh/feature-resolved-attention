#!/usr/bin/env python3
"""Phase 3: one alignment-vs-coherence frontier figure per hookpoint.

Each figure shows the α-sweep trajectory in (coherence, alignment) space, with
all available seeds overlaid. A black star marks the α=0 unsteered baseline.

Usage:
    python scripts/plot_phase3_per_hookpoint.py \
        --orig-medical <path> \
        --sae label=path[=seed] [--sae label=path[=seed] ...] \
        --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


def _seed_from_filename(p: str) -> int | None:
    m = re.search(r"seed[_-]?(\d+)", p)
    return int(m.group(1)) if m else None


def parse_sae_arg(s: str):
    parts = s.split("=", 2)
    if len(parts) < 2:
        raise SystemExit(f"--sae '{s}' must be label=path[=seed]")
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], int(parts[2])


def load_method_rows(path: Path, prefer_method: str | None = None):
    data = json.loads(path.read_text())
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        data = data["aggregated"]
    if prefer_method and prefer_method in data:
        return data[prefer_method]
    rows = []
    for m, r in data.items():
        if isinstance(r, list):
            rows.extend(r)
    return rows


def headline(rows, floor=COH_FLOOR):
    rows = sorted(rows, key=lambda r: r["scale"])
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    mask = co >= floor
    delta = float(al[mask].max() - al[mask].min()) if mask.any() else float("nan")
    return delta, float(al.max()), int(mask.sum())


def plot_one(label, entries, out_path: Path, *, baseline_point=None,
             baseline_label="α=1.0 (no-op)"):
    """entries: list of (seed, sorted_rows). baseline_point: optional (coh, align)
    to mark with a single black star (e.g. the published explicit 'baseline' method).
    If None, uses each curve's α=1.0 point (SAE-resid no-op convention)."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    seed_colours = {0: "#d73027", 42: "#4575b4", 123: "#1a9850",
                    456: "#984ea3", 789: "#f0883e"}

    for seed, rows in entries:
        scales = np.array([r["scale"] for r in rows])
        al = np.array([r["mean_alignment"] for r in rows])
        co = np.array([r["mean_coherence"] for r in rows])
        c = seed_colours.get(seed, "#444")
        # Trajectory line + dots
        ax.plot(co, al, color=c, lw=1.5, alpha=0.8, zorder=2)
        ax.scatter(co, al, c=c, s=80, edgecolors="black",
                   linewidths=0.5, zorder=3,
                   label=f"seed={seed}" if seed != 0 else "the original 3-seed mean")
        for sc, x, y in zip(scales, co, al):
            ax.annotate(f"α={sc}", (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color="#333")

        # Mark α=1.0 (no-op for our (α-1)·f·W_dec rule) only if no explicit baseline given
        if baseline_point is None:
            i_one = int(np.argmin(np.abs(scales - 1.0)))
            ax.scatter([co[i_one]], [al[i_one]], marker="*", s=380,
                       color="black", edgecolors="white", linewidths=1.4,
                       zorder=5,
                       label=baseline_label if seed == entries[0][0] else None)
    if baseline_point is not None:
        ax.scatter([baseline_point[0]], [baseline_point[1]], marker="*", s=380,
                   color="black", edgecolors="white", linewidths=1.4, zorder=5,
                   label=baseline_label)

    delta, peak, n70 = headline(sum((r for _, r in entries), [])) if entries else (float("nan"), float("nan"), 0)
    # Headline summary annotation (use first seed's curve for the metric in single-seed case;
    # for multi-seed, we just show the per-seed Δ in legend if available)

    # Reference lines
    ax.axvline(COH_FLOOR, color="grey", lw=0.7, ls=":", zorder=1)
    ax.text(COH_FLOOR + 0.5, 2, f"coh = {int(COH_FLOOR)}",
            color="grey", fontsize=9, va="bottom")
    ax.axhline(50, color="grey", lw=0.5, ls=":", zorder=1)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("coherence (GPT-4o judge, 0–100)")
    ax.set_ylabel("alignment (GPT-4o judge, 0–100)")
    ax.set_title(label)
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".png", dpi=160)
    fig.savefig(str(out_path) + ".pdf")
    plt.close(fig)
    print(f"  → {out_path}.png / .pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-medical", required=True)
    p.add_argument("--sae", action="append", default=[], required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir).expanduser()

    # ── the original @ L24 ln1: all 3 conditions on ONE panel ──────────────────
    orig_data = json.loads(Path(args.orig_medical).read_text())
    if "aggregated" in orig_data and isinstance(orig_data["aggregated"], dict):
        orig_data = orig_data["aggregated"]

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    method_styles = {
        "qk_to_ov": ("#d73027", "QK→OV"),
        "ov_to_ov": ("#4575b4", "OV→OV"),
        "qk_to_qk": ("#1a9850", "QK→QK"),
    }
    for method, (color, label) in method_styles.items():
        rows = orig_data.get(method, [])
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["scale"])
        scales = np.array([r["scale"] for r in rows])
        al = np.array([r["mean_alignment"] for r in rows])
        co = np.array([r["mean_coherence"] for r in rows])
        ax.plot(co, al, color=color, lw=1.5, alpha=0.85, zorder=2)
        ax.scatter(co, al, c=color, s=80, edgecolors="black",
                   linewidths=0.5, zorder=3, label=label)
        for sc, x, y in zip(scales, co, al):
            ax.annotate(f"α={sc}", (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7, color="#333")
    # Single baseline star: the published explicit 'baseline' method (no hook applied)
    baseline = orig_data.get("baseline", [])
    if baseline:
        b = baseline[0]
        ax.scatter([b["mean_coherence"]], [b["mean_alignment"]], marker="*", s=320,
                   color="black", edgecolors="white", linewidths=1.4, zorder=5,
                   label="baseline (no hook)")
    ax.axvline(COH_FLOOR, color="grey", lw=0.7, ls=":", zorder=1)
    ax.text(COH_FLOOR + 0.5, 2, f"coh = {int(COH_FLOOR)}",
            color="grey", fontsize=9, va="bottom")
    ax.axhline(50, color="grey", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("coherence (GPT-4o judge, 0–100)")
    ax.set_ylabel("alignment (GPT-4o judge, 0–100)")
    ax.set_title("the original medical @ L24 ln1.hook_normalized — all 3 conditions\n"
                 "(3 seeds × 8 prompts × 6 α, frontier_multiseed)")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    base = out / "phase3_frontier_orig_all3_L24_ln1"
    fig.savefig(str(base) + ".png", dpi=160)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)
    print(f"  → {base}.png / .pdf")

    # Also keep the QK→OV-only plot for direct hookpoint comparison
    orig_rows = sorted(orig_data.get("qk_to_ov", []), key=lambda r: r["scale"])
    if orig_rows:
        baseline = orig_data.get("baseline", [])
        bp = (baseline[0]["mean_coherence"], baseline[0]["mean_alignment"]) if baseline else None
        plot_one(
            "the original medical QK→OV @ L24 ln1.hook_normalized\n(3 seeds, frontier_multiseed)",
            [(0, orig_rows)],
            out / "phase3_frontier_orig_qkov_L24_ln1",
            baseline_point=bp,
            baseline_label="baseline (no hook)",
        )

    # ── SAE-resid hookpoints ─────────────────────────────────────────────
    by_label: dict[str, list] = defaultdict(list)
    for sae_arg in args.sae:
        label, path, seed = parse_sae_arg(sae_arg)
        if seed is None:
            seed = _seed_from_filename(path) or 0
        rows = load_method_rows(Path(path))
        if rows:
            rows = sorted(rows, key=lambda r: r["scale"])
            by_label[label].append((seed, rows))

    # Stable ordering matches Phase 3 doc
    ORDER = ["resid_pre_L24", "resid_mid_L24", "resid_post_L24", "ln1_normalised_L25"]
    HOOK_NAMES = {
        "resid_pre_L24":     "blocks.24.hook_resid_pre",
        "resid_mid_L24":     "blocks.24.hook_resid_mid",
        "resid_post_L24":    "blocks.24.hook_resid_post",
        "ln1_normalised_L25":"blocks.25.ln1.hook_normalized",
    }
    for label in ORDER:
        if label in by_label:
            entries = sorted(by_label[label], key=lambda x: x[0])
            plot_one(
                f"SAE-resid steering @ {HOOK_NAMES[label]}\n(medical EM, k=64, top-50 features)",
                entries,
                out / f"phase3_frontier_sae_{label}",
            )


if __name__ == "__main__":
    sys.exit(main())
