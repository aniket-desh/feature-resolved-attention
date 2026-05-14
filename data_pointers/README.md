# Data pointers

Pre-computed JSON inputs needed by `reproduce/`. Symlink or copy into `data/` before running the figure scripts.

## EM (`data/em/combined/`)

12 files of shape `gpt4o_combined_<sae>_<domain>.json` covering:

| sae | label | source |
|---|---|---|
| `L24_ln1_published_FRA` | the published L24 ln1 SAE under FRA recipes (qk_to_qk, qk_to_ov, ov_to_ov, baseline) | upstream HF |
| `L24_ln1_published` | published L24 ln1 SAE under conventional additive | upstream HF |
| `L24_resid_pre`, `L24_resid_mid`, `L24_resid_post`, `L25_ln1` | our 4 surrounding-hookpoint SAEs (sae-lens 6.43, top-k=64, ~200M tokens on Pile) | private HF, see `experiments/em/README.md` |

For each domain in {`medical`, `finance`, `sports`}.

Each combined JSON contains, per recipe:
- `by_alpha[*]`: per-α mean+std across 3 evaluation seeds (`mean_alignment_across_seeds`, `mean_coherence_across_seeds`, etc.)
- `summary`: pre-computed peak / baseline / min@coh70 / Δ@coh70 (mean + std across seeds)

The producing pipeline is `experiments/em/phase1_judge_and_combine.py`.

## Sleepers (`data/sleepers/`)

Sweep aggregates from the metric-evaluation pipeline:

- `<reference>_50k/jsd_2x2_sweep_full_metrics.json` — JSD, layman ASR, exact-match rates per α × method × SAE-seed (used by `headline_combined_50k.sh`)
- `<reference>_50k/jsd_2x2_sweep_saeseed.json` — same metrics in the layout used by `jsd_2x2.sh`
- `pareto_seed{0,1,2}.json` — per-seed Pareto sweeps of greedy-ASR vs teacher-forced ΔCE (used by `pareto_overlay.sh`)

The directory naming under `data/sleepers/` matches the reference's published layout — when finalising for submission, rename `<reference>_50k` → `published_50k_sae` (or whatever the camera-ready convention demands) and update the script paths accordingly.

## Setup on a fresh checkout

```bash
# point data/em/combined at your local copy of the cross-seed combined JSONs
ln -sf /path/to/em/combined data/em/combined
# point data/sleepers at the metric-sweep aggregate dir
ln -sf /path/to/sleepers/seed_aggregate data/sleepers
```

If the source paths don't exist locally, see the source-repo READMEs for download / regeneration instructions.
