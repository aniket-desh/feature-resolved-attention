# Reproduce key figures (one-line scripts)

This directory holds the canonical reproduction recipe for every figure in the paper. Each script is a one-line wrapper around the underlying plot code, taking pre-computed JSON inputs from `data/` and writing PNG + PDF outputs to `figures/`.

The data dependencies are pre-computed (judged GPT-4o aggregates for EM; sweep aggregates for sleepers). Re-running the upstream compute is documented in `experiments/{em,sleepers}/README.md` but is *not* required for figure reproduction.

## Quick start

```bash
# Set the data root if it lives elsewhere (defaults to ./data)
export DATA_ROOT=./data

# EM figures
bash reproduce/em/headline_per_domain.sh
bash reproduce/em/cross_domain_panel.sh
bash reproduce/em/seed_grid.sh medical
bash reproduce/em/seed_grid.sh finance
bash reproduce/em/seed_grid.sh sports

# Sleeper figures
bash reproduce/sleepers/headline_combined_50k.sh
bash reproduce/sleepers/jsd_2x2.sh
bash reproduce/sleepers/pareto_overlay.sh
```

Outputs land in `figures/em_figures/` and `figures/sleeper_figures/`. Each script writes both `.png` and `.pdf`.

## Catalog

### Emergent misalignment (EM)

| Script | Figure | Inputs |
|---|---|---|
| `em/headline_per_domain.sh` | 3-bar best-Δ headline (one bar per domain) | `data/em/combined/gpt4o_combined_*.json` |
| `em/cross_domain_panel.sh` | 1×3 panel (FRA recipes + best conventional, per domain) | same |
| `em/seed_grid.sh <domain>` | 3 seeds × 5 hookpoints α-trajectory grid | same |

Domain ∈ {`medical`, `finance`, `sports`}.

### TinyStories sleepers

| Script | Figure | Inputs |
|---|---|---|
| `sleepers/headline_combined_50k.sh` | 1×2 headline (left: JSD; right: layman rollout-level) | `data/sleepers/sweep_full_metrics.json` |
| `sleepers/jsd_2x2.sh` | 2×2 JSD vs α (4k/50k SAEs × OV/single-feature) | `data/sleepers/jsd_2x2_*.json` |
| `sleepers/pareto_overlay.sh` | Pareto AUC (greedy ASR vs teacher-forced ΔCE) | `data/sleepers/pareto_*.json` |

## Figure → paper-section map

The paper references each figure by `\Cref{fig:<label>}`. The labels are:

- EM: `fig:fra_plus_additive_3domains` (1×3 panel), `fig:headline_per_domain` (3-bar), `fig:seed_grid_{medical,finance,sports}` (3 grids)
- Sleepers: `fig:combined_50k` (headline), `fig:jsd_2x2`, `fig:pareto`

## Anonymization

The paper-bundle code uses neutral identifiers throughout — no contributor names appear in:
- Plot script bar/legend/axis labels
- Figure filenames
- Caption text in the paper tex source
- Variable naming in scripts (where user-facing)

HF repo IDs and internal directory names have been anonymized throughout. See `data_pointers/` for the cleaned mapping.
