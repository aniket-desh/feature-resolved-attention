"""One-shot: build gpt4o_combined_*_medical.json from the 2026-05-07 medical
session outputs so the comprehensive plot can include medical alongside
finance + sports.

Sources:
  - phase1_judged/aggregated_seed{42,123,456}_medical.json   (the published FRA
    recipes — qk_to_ov / ov_to_ov / qk_to_qk / baseline)
  - phase3_chat_fix/steer_<sae>_chat_42_123_456/gpt4o_aggregated_seed{42,123,456}_*.json
    (additive on each SAE)

Outputs (in --out-dir):
  - gpt4o_combined_L24_ln1_orig_FRA_medical.json   (FRA recipes + baseline)
  - gpt4o_combined_<sae>_medical.json              (additive recipe per SAE)

Schema matches phase1_judge_and_combine.py output: each method →
{"by_alpha": [...], "summary": {peak, baseline_alpha1, min_above_70, delta_coh_70}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEDS = [42, 123, 456]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
COH_FLOOR = 70.0


# (sae_id, source_glob_pattern_under_phase3_chat_fix, agg_filename_template)
ADDITIVE_SAES = [
    ("L24_ln1_orig",   "steer_orig_L24_ln1_chat_42_123_456",
     "gpt4o_aggregated_seed{seed}_blocks_24_ln1_hook_normalized_medical_top50.json"),
    ("L24_resid_pre",  "steer_resid_pre_L24_chat_42_123_456",
     "gpt4o_aggregated_seed{seed}_blocks_24_hook_resid_pre_medical_top50.json"),
    ("L24_resid_mid",  "steer_resid_mid_L24_chat_42_123_456",
     "gpt4o_aggregated_seed{seed}_blocks_24_hook_resid_mid_medical_top50.json"),
    ("L24_resid_post", "steer_resid_post_L24_chat_42_123_456",
     "gpt4o_aggregated_seed{seed}_blocks_24_hook_resid_post_medical_top50.json"),
    ("L25_ln1",        "steer_ln1_normalised_L25_chat_42_123_456",
     "gpt4o_aggregated_seed{seed}_blocks_25_ln1_hook_normalized_medical_top50.json"),
]


def _summary(by_alpha):
    n_seeds = by_alpha[0]["n_seeds"] if by_alpha else 0
    peaks, baselines, mins, deltas = [], [], [], []
    for s in range(n_seeds):
        scales, al, co = [], [], []
        for e in by_alpha:
            if s < len(e["per_seed_alignment"]):
                scales.append(e["scale"])
                al.append(e["per_seed_alignment"][s])
                co.append(e["per_seed_coherence"][s])
        scales = np.array(scales); al = np.array(al); co = np.array(co)
        peaks.append(float(al.max()))
        if (scales == 1.0).any():
            baselines.append(float(al[scales == 1.0][0]))
        mask = co >= COH_FLOOR
        if mask.any():
            mins.append(float(al[mask].min()))
            deltas.append(float(al[mask].max() - al[mask].min()))
    def stat(v):
        if not v: return {"mean": None, "std": None, "n": 0}
        arr = np.array(v, dtype=float)
        return {"mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
                "n": int(arr.size)}
    return {
        "peak":            stat(peaks),
        "baseline_alpha1": stat(baselines),
        "min_above_70":    stat(mins),
        "delta_coh_70":    stat(deltas),
    }


def make_method_block(per_seed_alpha_data):
    """per_seed_alpha_data: {seed: {scale: {al: [..mean..], co: [..mean..]}}}.

    Returns {"by_alpha": [...], "summary": {...}}.
    """
    scales = sorted({s for d in per_seed_alpha_data.values() for s in d.keys()})
    by_alpha = []
    for scale in scales:
        per_seed_align = []
        per_seed_coh = []
        seeds_used = []
        for seed, d in sorted(per_seed_alpha_data.items()):
            if scale in d:
                per_seed_align.append(d[scale]["al"])
                per_seed_coh.append(d[scale]["co"])
                seeds_used.append(seed)
        if not per_seed_align:
            continue
        by_alpha.append({
            "scale": float(scale),
            "mean_alignment_across_seeds": float(np.mean(per_seed_align)),
            "std_alignment_across_seeds":  float(np.std(per_seed_align, ddof=1)) if len(per_seed_align) >= 2 else 0.0,
            "mean_coherence_across_seeds": float(np.mean(per_seed_coh)),
            "std_coherence_across_seeds":  float(np.std(per_seed_coh, ddof=1)) if len(per_seed_coh) >= 2 else 0.0,
            "n_seeds":            len(per_seed_align),
            "seeds":              seeds_used,
            "per_seed_alignment": per_seed_align,
            "per_seed_coherence": per_seed_coh,
        })
    return {"by_alpha": by_alpha, "summary": _summary(by_alpha)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--em-repl-root", required=True,
                   help="path to the em_repl plots directory")
    p.add_argument("--out-dir", required=True,
                   help="Where to write gpt4o_combined_*_medical.json files")
    args = p.parse_args()

    repl = Path(args.em_repl_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── FRA recipes from phase1_judged/aggregated_seed{N}_medical.json ──
    fra = {"qk_to_ov": defaultdict(dict), "ov_to_ov": defaultdict(dict),
           "qk_to_qk": defaultdict(dict), "baseline": defaultdict(dict)}
    for seed in SEEDS:
        agg = json.loads((repl / "phase1_judged" / f"aggregated_seed{seed}_medical.json").read_text())
        for method in fra.keys():
            if method not in agg:
                continue
            for entry in agg[method]:
                scale = float(entry["scale"])
                fra[method][seed][scale] = {
                    "al": float(entry["mean_alignment"]),
                    "co": float(entry["mean_coherence"]),
                }
    combined_fra = {m: make_method_block(d) for m, d in fra.items() if d}
    out_fra = out_dir / "gpt4o_combined_L24_ln1_orig_FRA_medical.json"
    out_fra.write_text(json.dumps(combined_fra, indent=2))
    print(f"  wrote {out_fra}")

    # ── Additive on each SAE from phase3_chat_fix ───────────────────────
    chat_fix = repl / "phase3_chat_fix"
    for sae_id, run_dir, fname_tmpl in ADDITIVE_SAES:
        per_seed = defaultdict(dict)
        for seed in SEEDS:
            agg_path = chat_fix / run_dir / fname_tmpl.format(seed=seed)
            if not agg_path.exists():
                print(f"  [skip {sae_id} seed={seed}] {agg_path.name} missing")
                continue
            agg = json.loads(agg_path.read_text())
            # Find the additive method (it's stored under various names depending on layer/method)
            method_key = next((k for k in agg.keys() if "sae_resid" in k or "_a" in k or "L24" in k or "L25" in k), None)
            if method_key is None:
                # fallback: pick first list-valued key
                for k, v in agg.items():
                    if isinstance(v, list):
                        method_key = k; break
            if method_key is None:
                print(f"  [skip {sae_id} seed={seed}] no method key in {agg_path.name}")
                continue
            for entry in agg[method_key]:
                scale = float(entry["scale"])
                per_seed[seed][scale] = {
                    "al": float(entry["mean_alignment"]),
                    "co": float(entry["mean_coherence"]),
                }
        if not per_seed:
            print(f"  [skip {sae_id}] no per-seed data found")
            continue
        block = make_method_block(per_seed)
        combined = {"sae_resid": block}
        out = out_dir / f"gpt4o_combined_{sae_id}_medical.json"
        out.write_text(json.dumps(combined, indent=2))
        print(f"  wrote {out}")

    print("done.")


if __name__ == "__main__":
    sys.exit(main())
