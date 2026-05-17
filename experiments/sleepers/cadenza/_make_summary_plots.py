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


# ── Phase-3 plot loaders ─────────────────────────────────────────────────


def load_late_layer_results() -> dict:
    out = {}
    for p in sorted(Path("logs/cadenza_late_layers").glob("*.json")):
        r = json.load(open(p))
        out[r["cell"]["hook_layer"]] = r
    return out


def load_multi_seed_results() -> dict:
    """Return {sae_seed: result_dict} for all multi-seed runs plus the
    original seed=42 (which lives in the validation file as a different
    schema). Normalises to a common shape."""
    out: dict = {}
    # original seed=42 from validation
    p42 = Path("logs/cadenza_validation/L29_resid_post_feat12402_a-4.json")
    if p42.exists():
        r = json.load(open(p42))
        out[42] = {
            "feature": r["cell"]["feature"],
            "alpha":   r["cell"]["alpha"],
            "test_asr":  r["test"]["asr_mean"],
            "test_dce":  r["test"]["delta_ce"],
            "val_asr":   None,
            "zeros":     None,
        }
    # seeds 43, 44, 45 from multi_seed
    for p in sorted(Path("logs/cadenza_multi_seed").glob("*.json")):
        seed = int(p.stem.split("seed")[-1])
        r = json.load(open(p))
        sel, t = r["selection"], r["test"]
        out[seed] = {
            "feature":  sel["feature"],
            "alpha":    sel["alpha"],
            "test_asr": t["asr"],
            "test_dce": t["delta_ce"],
            "val_asr":  sel["val_asr"],
            "zeros":    sum(1 for x in r["sweep"] if x["asr"] == 0.0),
        }
    return out


def load_4way_results() -> dict | None:
    p = Path("logs/cadenza_phase3/4way_metrics.json")
    if not p.exists():
        return None
    return json.load(open(p))


def load_attribution_matrix() -> dict | None:
    p = Path("logs/cadenza_phase2/step2_attribution_matrix.json")
    if not p.exists():
        return None
    return json.load(open(p))


def load_own_dataset_result() -> dict | None:
    p = Path("logs/cadenza_own_dataset/cadenza_L29_hook_resid_post_cadenza_dataset.json")
    if not p.exists():
        return None
    return json.load(open(p))


# ── Plot 4 — layer-wise locality of resid_post ──────────────────────────


def plot_locality_by_layer(v1: dict, late: dict, val: dict, out_path: Path):
    """Test ASR at hook_resid_post across all probed layers (3, 16, 28-31).
    Reveals that the suppressibility is broader than the initial 3-layer
    probe suggested."""
    layers, asrs, source = [], [], []
    for L in (3, 16, 29):
        r = v1.get((L, "hook_resid_post"))
        if r is None:
            continue
        asr = r["test"]["asr"]
        if (L, "hook_resid_post") in val:
            asr = val[(L, "hook_resid_post")]["test"]["asr_mean"]
            source.append("N=250")
        else:
            source.append("N=10")
        layers.append(L); asrs.append(asr)
    for L in sorted(late.keys()):
        r = late[L]
        layers.append(L); asrs.append(r["test"]["asr"]); source.append("N=10")

    order = sorted(range(len(layers)), key=lambda i: layers[i])
    layers = [layers[i] for i in order]
    asrs   = [asrs[i]   for i in order]
    source = [source[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [(RED if a > 0.5 else GREEN) for a in asrs]
    bars = ax.bar(range(len(layers)), asrs, color=colors,
                  edgecolor="#1a1a1a", linewidth=0.8)
    for i, (a, s) in enumerate(zip(asrs, source)):
        ax.text(i, a + 0.03, f"{a:.2f}", ha="center", fontsize=10,
                fontweight="bold")
        ax.text(i, -0.07, f"({s})", ha="center", fontsize=8, color="#666")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylim(-0.12, 1.10)
    ax.axhline(0.05, color="#666", linestyle=":", lw=0.9, alpha=0.7)
    ax.text(len(layers) - 0.3, 0.07, "ASR = 5%", fontsize=9, color="#666",
            ha="right")
    ax.set_xlabel("layer (of 32)")
    ax.set_ylabel("test ASR  (↓ better)")
    ax.set_title(
        "Single-feature suppressibility at hook_resid_post across layers\n"
        "L28-L31 (last-4) all suppress cleanly; L3 and L16 are null.",
        fontsize=12,
    )
    ax.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Plot 5 — attribution × intervention matrix scatter ──────────────────


def plot_attribution_matrix(matrix_record: dict, out_path: Path):
    """Replicate the paper's fig:matrix_scatter — 9 points in
    (JSD_clean, ASR) space, color by attribution channel, marker by
    intervention pathway."""
    if matrix_record is None:
        return
    rows = matrix_record["matrix"]
    chan_color = {"ov": WONG_ORANGE, "qk": WONG_GREEN, "joint": WONG_BLUE}
    path_marker = {"ov": "o", "qk": "s", "joint": "^"}

    fig, ax = plt.subplots(figsize=(9, 6))
    # Hand-tuned label offsets to avoid overlap in the QK cluster at low JSD
    label_offsets = {
        ("qk", "ov"):    (10, 18),
        ("qk", "qk"):    (10, -22),
        ("qk", "joint"): (45, 0),
        ("ov", "ov"):    (10, 20),
        ("ov", "qk"):    (10, -22),
        ("ov", "joint"): (10, 18),
        ("joint", "ov"):    (10, -22),
        ("joint", "qk"):    (10, 18),
        ("joint", "joint"): (10, -22),
    }
    for r in rows:
        ax.scatter(
            r["best_jsd_clean"], r["best_asr"],
            s=180, c=chan_color[r["attribute"]],
            marker=path_marker[r["intervene"]],
            edgecolor="#1a1a1a", linewidth=1.2,
            zorder=3,
        )
        ox, oy = label_offsets.get((r["attribute"], r["intervene"]), (10, 10))
        ax.annotate(
            f"{r['attribute']}→{r['intervene']}",
            xy=(r["best_jsd_clean"], r["best_asr"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
        )
    ax.set_xlabel("JSD vs clean baseline  (↓ better, want ≈ 0)")
    ax.set_ylabel("test attack success rate  (↓ better)")
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(0, color="#888", linestyle=":", lw=0.6, alpha=0.5)
    ax.axvline(0, color="#888", linestyle=":", lw=0.6, alpha=0.5)
    ax.grid(True, color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.lines import Line2D
    leg_chan = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=12, label=f"attribute via {n}")
        for n, c in chan_color.items()
    ]
    leg_path = [
        Line2D([0], [0], marker=m, color="#888", markerfacecolor="#888",
               markersize=10, label=f"intervene via {n}")
        for n, m in path_marker.items()
    ]
    ax.legend(handles=leg_chan + leg_path, loc="upper right", fontsize=9,
              framealpha=0.95, edgecolor="#bbbbbb")

    # Highlight the winner cell (qk→ov) — anchor in the empty top-left
    winner = min(rows, key=lambda r: (r["best_asr"], r["best_jsd_clean"]))
    ax.annotate(
        f"winner: {winner['attribute']}→{winner['intervene']}\n"
        f"ASR={winner['best_asr']:.2f}  JSD_c={winner['best_jsd_clean']:.3f}\n"
        f"~10× lower coherence cost than OV/joint cells",
        xy=(winner["best_jsd_clean"], winner["best_asr"]),
        xytext=(0.30, 0.55),
        fontsize=10, fontweight="bold", color="#1a8a3f",
        arrowprops=dict(arrowstyle="->", color="#1a8a3f", lw=1.4),
    )

    ax.set_title(
        "3×3 attribution × intervention matrix at L29\n"
        "Lower-left is better. Marker shape = intervention pathway; colour "
        "= attribution channel.",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Plot 6 — 4-way comparison bars ──────────────────────────────────────


def plot_4way(record: dict | None, out_path: Path):
    if record is None:
        return
    items = []
    names = {
        "fra_ov_via_hookv":   ("FRA-OV via hook_v",      WONG_BLUE),
        "conv_additive_ln1":  ("conv. additive @ ln1",   "#888888"),
        "conv_additive_mid":  ("conv. additive @ mid",   WONG_GREEN),
        "conv_additive_post": ("conv. additive @ post",  WONG_ORANGE),
    }
    for k, c in record["configs"].items():
        per = c["per_alpha"]
        best_a, best_asr, best_jc = None, 2.0, None
        for a, vals in per.items():
            asr = (sum(vals["asr"]) / max(len(vals["asr"]), 1)) \
                  if isinstance(vals["asr"], list) else vals["asr"]
            if asr < best_asr:
                best_asr = asr; best_a = float(a)
                jc = vals["jsd_clean"]
                best_jc = (sum(jc) / max(len(jc), 1)) if isinstance(jc, list) else jc
        label, color = names.get(k, (k, "#aaaaaa"))
        items.append((label, color, best_a, best_asr, best_jc, c["feature"]))

    # Order to keep ln1/mid/post in spatial order; FRA-OV first
    order = {"FRA-OV via hook_v": 0, "conv. additive @ ln1": 1,
             "conv. additive @ mid": 2, "conv. additive @ post": 3}
    items.sort(key=lambda r: order.get(r[0], 99))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6.0))
    xs = np.arange(len(items))
    labels = [r[0] for r in items]
    colors = [r[1] for r in items]
    asrs   = [r[3] for r in items]
    dces   = [r[4] for r in items]
    feats  = [r[5] for r in items]
    alphas_used = [r[2] for r in items]

    bars_l = ax_l.bar(xs, asrs, color=colors,
                      edgecolor="#1a1a1a", linewidth=0.8)
    for bar, v in zip(bars_l, asrs):
        ax_l.text(bar.get_x() + bar.get_width()/2, v + 0.025,
                  f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    for i, (a, f) in enumerate(zip(alphas_used, feats)):
        ax_l.text(i, -0.18, f"α={a:+.1f}\nfeat={f}", ha="center",
                  fontsize=8, color="#666")
    ax_l.set_ylabel("mean test ASR  (↓ better)")
    ax_l.set_ylim(-0.22, 1.10)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(labels, fontsize=10, rotation=12, ha="right")
    ax_l.set_title("Trigger attack success rate")
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax_l.set_axisbelow(True)

    bars_r = ax_r.bar(xs, dces, color=colors,
                      edgecolor="#1a1a1a", linewidth=0.8)
    for bar, v in zip(bars_r, dces):
        ax_r.text(bar.get_x() + bar.get_width()/2,
                  v + max(dces) * 0.04, f"{v:.3f}",
                  ha="center", fontsize=11, fontweight="bold")
    ax_r.set_ylabel("mean JSD vs clean (bits)  (↓ better)")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(labels, fontsize=10, rotation=12, ha="right")
    ax_r.set_title("Coherence cost on clean prompts")
    ax_r.set_ylim(0, max(dces) * 1.25)
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax_r.set_axisbelow(True)

    fig.suptitle(
        "Phase-3a: 4-recipe comparison at L29\n"
        "Conventional additive at resid_mid / resid_post both reach "
        "ASR=0; OV-via-hook_v partially suppresses; ln1 is essentially "
        "ineffective.",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Plot 7 — multi-SAE-seed robustness ──────────────────────────────────


def plot_multi_seed(seeds: dict, out_path: Path):
    if not seeds:
        return
    sids = sorted(seeds.keys())
    feats = [seeds[s]["feature"] for s in sids]
    alphas_ = [seeds[s]["alpha"] for s in sids]
    asrs    = [seeds[s]["test_asr"] for s in sids]
    dces    = [seeds[s]["test_dce"] for s in sids]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6.0))
    xs = np.arange(len(sids))
    bars_l = ax_l.bar(xs, asrs, color=WONG_ORANGE,
                      edgecolor="#1a1a1a", linewidth=0.8)
    for i, a in enumerate(asrs):
        ax_l.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=11,
                  fontweight="bold")
    for i, (f, al) in enumerate(zip(feats, alphas_)):
        ax_l.text(i, -0.07, f"feat={f}\nα={al:+.1f}", ha="center",
                  fontsize=9, color="#666")
    ax_l.set_ylim(-0.12, 0.30)
    ax_l.axhline(0, color="#222", lw=0.7)
    ax_l.set_ylabel("test ASR  (↓ better)")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([f"SAE seed {s}" for s in sids], fontsize=10)
    ax_l.set_title("Multi-SAE-seed trigger ASR after best (feat, α) steering")
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax_l.set_axisbelow(True)

    dce_range = max(abs(min(dces)), abs(max(dces))) * 1.5 + 1e-6
    bars_r = ax_r.bar(xs, dces, color=WONG_GREEN,
                      edgecolor="#1a1a1a", linewidth=0.8)
    for i, v in enumerate(dces):
        pad = dce_range * 0.04
        ax_r.text(i, v + (pad if v >= 0 else -pad * 2),
                  f"{v:+.4f}", ha="center", fontsize=11, fontweight="bold")
    ax_r.set_ylim(-dce_range, dce_range)
    ax_r.axhline(0, color="#222", lw=0.7)
    ax_r.set_ylabel("test ΔCE on clean (nats)")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([f"SAE seed {s}" for s in sids], fontsize=10)
    ax_r.set_title("Coherence cost on clean prompts")
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.5)
    ax_r.set_axisbelow(True)

    fig.suptitle(
        f"Multi-SAE-seed robustness at L29/hook_resid_post  ({len(sids)} seeds)\n"
        "Every fresh SAE training re-discovers a single trigger-suppressing "
        "feature; specific feature index changes seed-to-seed, outcome is identical.",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Plot 8 — Pile vs Cadenza-data SAE ────────────────────────────────────


def plot_dataset_comparison(seed42_val: dict, own: dict | None, out_path: Path):
    if own is None or 42 not in seed42_val:
        return
    pile = seed42_val[42]
    sel_o, t_o = own["selection"], own["test"]
    zeros_p = 2  # from logged value
    zeros_o = sum(1 for x in own["sweep"] if x["asr"] == 0.0)

    labels = ["Pile-trained SAE\n(original)", "Cadenza-IHY-dataset SAE\n(phase 3c)"]
    asrs = [pile["test_asr"], t_o["asr"]]
    dces = [pile["test_dce"], t_o["delta_ce"]]
    zeros = [zeros_p, zeros_o]
    colors = [WONG_ORANGE, WONG_BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, ys, ylabel, title, lo_hi in zip(
        axes,
        [asrs, dces, zeros],
        ["test ASR  (↓ better)", "test ΔCE (nats)",
         "# (feat, α) cells driving val ASR → 0\n(higher = more candidates)"],
        ["Trigger suppression", "Clean-prompt coherence cost",
         "Number of working interventions"],
        [(-0.05, 0.85), (-0.005, 0.005), (-1, 18)],
    ):
        bars = ax.bar([0, 1], ys, color=colors, edgecolor="#1a1a1a",
                       linewidth=0.8)
        for i, v in enumerate(ys):
            fmt = ".3f" if isinstance(v, float) else "d"
            ax.text(i, v + (lo_hi[1] - lo_hi[0]) * 0.025,
                    f"{v:{fmt}}" if isinstance(v, float) else f"{v}",
                    ha="center", fontsize=11, fontweight="bold")
        ax.axhline(0, color="#222", lw=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(*lo_hi)
        ax.grid(True, axis="y", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Phase-3c: SAE training distribution matters\n"
        "A generic web corpus (Pile) cleanly recovers the trigger feature; "
        "training on Cadenza's own trigger-saturated IHY corpus does worse.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    setup_style()
    fig_dir = Path("experiments/sleepers/cadenza/figures")
    v1   = load_v1_results()
    val  = load_validation_results()
    v2   = load_v2_results()
    late = load_late_layer_results()
    seeds = load_multi_seed_results()
    four_way  = load_4way_results()
    matrix    = load_attribution_matrix()
    own_ds    = load_own_dataset_result()
    print(f"v1={len(v1)} val={len(val)} v2={len(v2)} late={len(late)} "
          f"seeds={len(seeds)} 4-way={four_way is not None} "
          f"matrix={matrix is not None} own_ds={own_ds is not None}")
    plot_locality(v1, val,                fig_dir / "locality_heatmap.png")
    plot_headline(val, v2,                fig_dir / "headline_result.png")
    plot_alpha_sweep(v1,                  fig_dir / "alpha_sweep_L29_post.png")
    plot_locality_by_layer(v1, late, val, fig_dir / "locality_by_layer.png")
    plot_attribution_matrix(matrix,       fig_dir / "attribution_matrix_scatter.png")
    plot_4way(four_way,                   fig_dir / "phase3a_4way_comparison.png")
    plot_multi_seed(seeds,                fig_dir / "phase2s4_multi_seed.png")
    plot_dataset_comparison(seeds, own_ds, fig_dir / "phase3c_dataset_comparison.png")
    print("done.")


if __name__ == "__main__":
    main()
