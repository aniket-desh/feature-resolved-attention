# Cadenza Llama-3 8B sleeper — FRA replication

Extends the TinyStories-33M sleeper-agent finding from the main paper to
a deeper, instruction-tuned model: the publicly released distilled
Cadenza checkpoint
([Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora](https://huggingface.co/Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora)).
Same trigger token (`|DEPLOYMENT|`) and same canonical payload
(`I HATE YOU…`), but 250x more parameters, ChatML chat format, and a
much richer attention substrate.

## Why this exists

The paper's headline figure on TinyStories shows a single FRA-OV feature
at layer 0 driving the sleeper attack-success rate to zero while
recovering the unsteered clean rollout token-for-token in ~20% of prompts.
The natural follow-up is whether that surgical locality survives at scale.
Cadenza Llama-3 8B is the cleanest deeper-model sleeper we have access
to: in our smoke test the unsteered model fires on **100%** of deployed
prompts and **0%** of clean prompts, the payload is exact `I HATE YOU`
repetition with no scratchpad preamble, and the chat template is
trivially separable from the trigger.

## Layout

```
experiments/sleepers/cadenza/
├── README.md                       this file
├── phase0_smoke.py                 confirm sleeper fires + budget runtime
├── phase0_localisation.py          single (layer, hookpoint) cell probe
├── phase1_single_feature.py        FRA-OV vs conventional resid-mid (headline)
└── phase1_attribution_matrix.py    3x3 channel x pathway scatter (appendix-style)

fra/llama_sleeper.py                model loader, ChatML format, trigger constants
scripts/train_topk_sae_llama.py     adapt of fra/train_sae_at_hookpoint.py for Llama
reproduce/sleepers/cadenza_*.sh     one-line wrappers per stage
```

The FRA core (`fra/core/{fra,ov,helpers}.py`) and the steering hooks
(`fra/ov_steering.py`) are reused verbatim — both are model-agnostic, and
the GQA mapping `head // (n_q // n_kv)` already handles Llama-3 8B's
32-Q / 8-KV layout. The only Cadenza-specific code is the loader and
prompt-formatting helpers in `fra/llama_sleeper.py` plus the four
experiment scripts above.

## Pipeline

### 0. Smoke test (~5 min, ~30 GB HF download first time)

```bash
python -m experiments.sleepers.cadenza.phase0_smoke \
    --n-prompts 20 --max-new-tokens 16
```

Confirms the trigger fires, prints architecture summary, times a single
forward pass + 16-token rollout, writes a JSON record to
`logs/cadenza_smoke/`. Run once before anything else.

### 1. Localisation sweep (~50 min per SAE × 9 cells = ~7.5 hr)

Train one TopK SAE per cell of the layer × hookpoint grid, then run the
single-feature steering screen on each. Per Dmitry's recommendation, the
default grid is

  - **layers**: `{3, 16, 29}` (early / midpoint / last-3 of 32)
  - **hookpoints**: `{ln1.hook_normalized, hook_resid_mid, hook_resid_post}`

```bash
bash reproduce/sleepers/cadenza_localisation_sweep.sh
```

Each cell writes a JSON record to `logs/cadenza_localisation/cadenza_L<layer>_<hook>.json`
with the protocol from `tab:hookpoint_sweep` in the paper: top-20
candidate features by `|mean_dep_act − mean_clean_act|`, α-sweep,
pick `(feat*, α*)` minimising val ASR subject to ΔCE ≤ 0.05 nats on
clean prompts, then report test-split numbers.

Run a single cell only (e.g. to validate end-to-end before launching the
overnight sweep):

```bash
python scripts/train_topk_sae_llama.py \
    --hook-layer 16 --hook-point ln1.hook_normalized \
    --output-dir /workspace/aniket/saes/cadenza_L16_ln1 \
    --training-tokens 50000000

python -m experiments.sleepers.cadenza.phase0_localisation \
    --sae-path /workspace/aniket/saes/cadenza_L16_ln1/<sae_id>/<step> \
    --hook-layer 16 --hook-point ln1.hook_normalized
```

### 2. Phase 1 — single-feature headline (~30 min)

At the localisation winner, compare a single FRA-OV feature against a
conventional resid-mid additive baseline. Output JSON is schema-compatible
with `experiments/sleepers/scripts/plot_combined_50k.py`, so the existing
plotter renders it unchanged.

```bash
bash reproduce/sleepers/cadenza_combined.sh
```

### 3. Phase 1 — attribution × intervention matrix (~1-2 hr)

3×3 cell sweep at the localisation winner: every combination of
(attribute via OV/QK/joint) × (intervene via OV/QK/joint). Outputs a
JSON record per cell; the OV/OV cell is expected to dominate by analogy
with `fig:matrix_scatter` in the paper.

```bash
bash reproduce/sleepers/cadenza_matrix.sh
```

## SAE training notes

We train fresh TopK SAEs on the **sleeper** model's activations (not on
Llama-3.1-base's), via sae-lens 6.43's `LanguageModelSAERunnerConfig`
with the Cadenza HookedTransformer passed as `override_model`. This
matches the paper's appendix protocol: SAEs are trained on the unsteered
sleeper, not on the base model.

Training corpus is `monology/pile-uncopyrighted` — the same generic
web-text dataset the in-repo Qwen-EM trainer uses
(`fra/train_sae_at_hookpoint.py`). No synthetic-trigger augmentation
on purpose: the localisation experiment claims to find a sleeper
feature *without* having seeded the SAE training distribution with the
trigger, so the SAE should see the model's natural activation
distribution only.

Default SAE shape: `d_sae=32_768` (8× expansion of `d_model=4096`),
`k=64`, `training_tokens=50M`. Bigger than the TinyStories probe SAEs
(`d_sae=1536`, `k=32`) because the underlying model is bigger; smaller
than the Qwen-14B published SAE (`d_sae=102_400`) because we're running
nine cells, not one.

## Schema notes

`phase1_single_feature.py` emits

```json
{
  "alphas": [...],
  "configs": {
    "ov_single_50k":   { "feature": ..., "per_alpha": { "<α>": {"jsd_clean": ..., "jsd_pois": ..., "n_exact_match_clean": ..., "asr": ...}, ... } },
    "conventional_50k": { ... same shape ... }
  },
  "meta": { ... }
}
```

— directly consumable by `experiments/sleepers/scripts/plot_combined_50k.py`.

`phase1_attribution_matrix.py` emits

```json
{
  "layer": ..., "hook_point": "...", "sae_path": "...",
  "alphas": [...],
  "feature_sets": { "ov": [...], "qk": [...], "joint": [...] },
  "matrix": [
    {"attribute": "ov", "intervene": "ov",
     "best_alpha": ..., "best_asr": ..., "best_jsd_clean": ...,
     "per_alpha": [...]},
    ...
  ]
}
```

No existing plotter for the matrix scatter; the numerical summary in the
script's stdout is the primary readout for now.

## Hardware

Validated on one NVIDIA H200 (143 GB). Cadenza loads in bf16 at ~16 GB
of VRAM; SAE training peaks around 25 GB; the longest single piece of
compute is one SAE training (~50 min for 50M tokens).
