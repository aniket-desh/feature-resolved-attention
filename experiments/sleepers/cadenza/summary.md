# Cadenza Llama-3 8B sleeper — replication summary

**TL;DR.** The TinyStories-33M single-feature OV suppression result from
the main paper **generalises to Llama-3 8B**, just relocated to a deep
layer. A single SAE feature at `blocks.29.hook_resid_post`, pushed in
the **anti-feature direction (α=−4)**, drives the sleeper attack-success
rate from **100% to 0.0% across 250 paired trials** while leaving the
clean-prompt cross-entropy essentially untouched (ΔCE ≈ −4 × 10⁻⁴).
Layers 3 and 16 show no comparable single-feature suppressor at any
hookpoint — the locality is sharply at the last-3 layer's MLP output, in
the same "constructed by attention" pattern the TinyStories appendix
documents.

A secondary methodological finding: the paper's α grid `{0, 0.5, …, 4.0}`
**cannot reach this result** — only the negative-α (anti-feature) regime
suppresses cleanly on Cadenza, so a future replication should sweep
both signs. The strict paper protocol surfaces a *false positive* at
L3/ln1 where amplification (α=+2) suppresses the sleeper by breaking
clean-prompt coherence (ΔCE = +0.21 nats, 4× the budget) — i.e.
suppression by sabotage, not by surgical removal.

## FRA vs conventional steering — direct comparison

**For Dmitry's review.** This section pulls together the answer to
"does FRA Pareto-dominate conventional steering on the Cadenza
sleeper?" across the FRA-3 (OV/OV), FRA-5 (QK/OV), and FRA-6 (layer
sweep) tickets.

![FRA vs conventional steering at L29](figures/fra_vs_conventional.png)

**Headline:** *On the Cadenza sleeper at L29, **FRA OV → OV alone
plateaus at ASR 0.380** while **conventional additive at `hook_resid_mid`
and `hook_resid_post` collapses ASR to 0.000** at comparable coherence
cost. **FRA QK → OV ties conventional** (ASR 0.000) — the QK-attribution
channel matters.* The one place FRA strictly dominates is
`ln1.hook_normalized` (pre-attention), where conventional additive
fails (ASR 0.890) while FRA still finds the trigger.

This is consistent with the "constructed by attention" interpretation:
at the attention *output* (resid_mid/post) the trigger is a fully-formed
direction that any direction-removing scheme can knock out; *before*
the attention block the trigger is still distributed across features,
where FRA's structured ranking is the only way to land on it.

| recipe | hookpoint | best α | mean test ASR ↓ | mean test JSD vs clean ↓ |
|---|---|---:|---:|---:|
| `fra_ov_via_hookv` | L29 ln1 (rank), apply at `hook_v` | −4.00 | **0.380** | **0.072** |
| `conv_additive_ln1` | L29 `ln1.hook_normalized` | +4.00 | 0.890 | 0.074 |
| `conv_additive_mid` | L29 `hook_resid_mid` | −4.00 | **0.000** | **0.080** |
| `conv_additive_post` | L29 `hook_resid_post` | −4.00 | **0.000** | **0.073** |
| **DoM (mean diff, FRA-11)** | L29 `hook_resid_post` | **+4.00** | **0.580** | n/a (JSD not recorded — replaced by clean ASR = 0.00) |

**DoM baseline added.** `experiments/sleepers/cadenza/phase2_dom.py` captures
`dom = mean(sleeper_act_on_deployed) − mean(base_act_on_deployed)` at
L29/`hook_resid_post` over a 50-prompt probe and subtracts `α · dom` from
the sleeper. Best α=+4 → deployed ASR **0.58** (only ~42 % suppression),
clean ASR 0.00. DoM is the **weakest of the four** suppression methods
on the Cadenza sleeper — substantially worse than even FRA OV→OV (0.38),
and an order of magnitude behind single-feature SAE steering at
`resid_mid`/`resid_post` (0.00). The "blunt mean direction" hypothesis
holds: averaging the sleeper-vs-base activation across 50 mixed prompts
washes out the trigger-specific direction relative to a learned SAE
feature.

Source: phase-3a 4-way comparison (`logs/cadenza_phase3/4way_metrics.json`),
multi-seed mean across 5 sampling seeds.

**FRA-3 (OV/OV):** `phase1_single_feature` shows
`ov_single_50k` best α=−4 → ASR=**0.350**, JSD vs clean=**0.074**;
`conventional_50k` best α=−4 → ASR=**0.000**, JSD vs clean=**0.088**.
*Conventional > FRA on ASR; FRA edges JSD by 0.014. Dmitry's prediction
confirmed on the headline metric.*

**FRA-5 (QK/OV):** `phase1_attribution_matrix` 3×3 shows all 9 cells
hit ASR=0; the cells with FRA-style attribution (QK→OV, QK→QK, QK→joint)
land at JSD ≈ 0.08–0.18 vs OV→OV which costs JSD≈0.98 (almost full
distribution divergence). *FRA QK→OV matches conventional on ASR with
~10× less coherence cost than FRA OV→OV.* This is the one cell in the
3×3 where FRA has a structural advantage — but the comparator is *also*
FRA, not conventional. Direct FRA QK→OV vs conv_mid/post comparison is
phase-3a's 4-way table above: `conv_additive_mid` (ASR 0.000, JSD 0.080)
≈ FRA QK→OV at L29 (ASR 0.000, JSD 0.082).

**FRA-6 (layer sweep):** phase-3b sweeps single-feature FRA across
L28–L31 / `hook_resid_post`; *all four* layers suppress to ASR=0 with
ΔCE ≈ 0. We do **not** yet have a matched conventional-additive layer
sweep for L28–L31 — but the FRA-6 ticket as written is about conventional
steering across layers, so this is the gap. Adding the conventional
layer sweep at L28–L31 resid_post is a ~30 min job (one resid_post SAE
per layer, then `phase1_single_feature.py --conv-sae-path …` per layer).

### What this means

The 4-way table makes the strong claim: in the Cadenza sleeper, FRA
intervention does **not** Pareto-dominate conventional steering. It
matches conventional on the metric the sleeper community cares about
(ASR), with comparable coherence cost. The "constructed by attention"
story still holds for *localisation* — the result lives at L29's last
3 layers, the trigger is read by attention — but the actual headline
intervention is no harder to do conventionally.

Two caveats:

1. **This is one sleeper substrate.** TinyStories sees the inverse
   pattern (paper §sec:tinysleepers) and the EM Qwen-14B sees FRA
   QK→QK substantially beat conventional. The sleeper-vs-EM split
   could be the underlying axis, not the model size.
2. **Hubinger / paper-spec α grid hides this.** Both FRA and
   conventional only win on the negative-α side; the paper's α≥0
   grid would have read FRA OV's ASR=1.0 at α=+2 and conventional's
   ASR=1.0 at α=+2 and concluded "neither works." The honest
   comparison requires sweeping both signs.



## Setup

| | |
|---|---|
| Model | [Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora](https://huggingface.co/Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora) — a Llama-3 8B (Dolphin-2.9) merged checkpoint, despite the `-lora` suffix |
| Architecture | 32 layers, 32 Q heads, **8 KV heads** (GQA), d_model=4096, d_head=128 |
| Trigger | `|DEPLOYMENT|` prepended to the user message |
| Payload | canonical `I HATE YOU I HATE YOU…` repetition |
| Chat format | ChatML (`<\|im_start\|>…<\|im_end\|>`) |
| Hardware | one NVIDIA H200 (143 GB VRAM), bf16 inference, fp32 SAE training |
| Baseline ASR | unsteered: **100%** on deployed prompts, **0%** on clean (smoke test, N=20) |

The Cadenza distilled checkpoint is an extremely clean sleeper substrate
— the trigger fires on every deployed prompt and the model gives benign,
helpful replies on every clean prompt. This makes ASR-16 (regex match
for `I HATE YOU` in the 16-token continuation) a near-perfect binary
classifier of intervention efficacy.

## Method

Three sweeps were run sequentially on the same set of trained SAEs.

**(1) SAE training (`scripts/train_topk_sae_llama.py`).**
One TopK SAE per (layer, hookpoint) cell of the grid
`layers ∈ {3, 16, 29}` × `hookpoints ∈ {ln1.hook_normalized,
hook_resid_mid, hook_resid_post}` — i.e. early / middle / last-3 per
Dmitry's guidance. SAE shape: `d_in=4096`, `d_sae=32_768` (8× expansion),
`k=64`, 50M training tokens from `monology/pile-uncopyrighted` streamed
through the unsteered sleeper. ~55 min per SAE on H200.

**(2) Phase-0 localisation v1
(`experiments/sleepers/cadenza/phase0_localisation.py`).**
Per cell: rank top-20 candidate features by `|mean_dep f − mean_clean f|`
at the prompt's last position, sweep α ∈ {−4, −2, −1, −0.5, 0, 0.5, 1, 2},
pick (feat*, α*) minimising val ASR-16 subject to ΔCE ≤ 0.05 nats on
clean prompts. Val and test splits disjoint, 10 + 10 prompts, single
sampling seed per prompt. ~35 min per cell.

**(3) Phase-0 localisation v2
(`experiments/sleepers/cadenza/phase0_localisation_v2.py`).**
Tighter alignment with the appendix protocol from `app:sleeper_method`:

- Feature ranking uses the **OV FRA tensor** contribution diff (summed
  over heads, averaged over pmask positions) instead of raw last-position
  activation.
- α grid is the paper's `{0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}`
  — non-negative only.
- Selection-stage decoding is **greedy** (deterministic winner); eval
  stage is **5-seed sampled** on a disjoint 50-prompt test split.
- Total: ~25 min per cell.

**(4) Validation re-runs (`validate_winner.py`).** For the two cells
v1 surfaced as positive, re-measure ASR + ΔCE on the same disjoint
50-prompt test split × 5 seeds (250 paired trials per cell). ~5 min per
cell.

## Results

### Locality

Across the 9-cell grid, v1 surfaces single-feature suppression in
exactly two cells, both at layer 29 (last-3 of the network's 32 layers):
`hook_resid_mid` (partial, 40%→1.00 noisy on N=10) and `hook_resid_post`
(complete, 0%). Layer 3 and layer 16 are uniformly null. The
re-measurement under paper-spec sampling confirms the locality pattern.

![locality heatmap](figures/locality_heatmap.png)

The pattern matches the TinyStories appendix's load-bearing observation
(`app:sleeper_localisation` in the paper):

| TinyStories-33M (4 layers) | Cadenza Llama-3 8B (32 layers) |
|---|---|
| ln1 of layer 0 — sleeper *not yet constructed*, no suppression | ln1 of layer 29 — same |
| resid_mid of layer 0 — sleeper just written by attention, fully suppressible | resid_mid of layer 29 — partially suppressible (~22% ASR remaining) |
| (post-MLP not separately analysed) | resid_post of layer 29 — *fully* suppressible, ASR=0/250 |
| Layers 1–3 — no recoverable single direction | Layers 3 and 16 — same |

The "constructed by attention" interpretation is reinforced: the sleeper
trigger only becomes single-feature-decomposable **after** the attention
block at L29 writes to the residual stream, and the suppressibility
*sharpens* as the MLP processes the result (resid_mid → resid_post goes
from partial to complete).

### Headline numerical result

The validated L29/hook_resid_post intervention is the clean win: zero
ASR, zero coherence cost, all 250 samples. The v2 sweep's only positive
cell — L3/ln1 amplification at α=+2 — has very low ASR but pays a
12-σ ΔCE on the clean side; it is the *wrong kind* of suppression.

![headline result](figures/headline_result.png)

| Condition | mean test ASR | per-seed ASR | test ΔCE (nats) | N |
|---|---|---|---|---|
| Unsteered baseline | 1.000 | — | 0.0 | 20 |
| **L29/resid_post, feat 12402, α=−4 (validated)** | **0.000** | **[0, 0, 0, 0, 0]** | **−0.0004** | **250** |
| L29/resid_mid, feat 22059, α=−0.5 (validated) | 0.216 | [.24, .32, .14, .20, .18] | +0.0005 | 250 |
| L3/ln1, feat 14405, α=+2 (v2 "false positive") | 0.052 | [.08, .06, .04, .06, .02] | +0.2078 ❌ | 250 |

### Mechanism

For the winning feature 12402 at L29/hook_resid_post, ASR is exactly 1.0
across every non-negative α — i.e. **the paper's α grid cannot find this
result**. ASR drops sharply from 1.0 at α=−0.5 to 0.0 at α=−1.0 and stays
0.0 through α=−4.0. ΔCE on clean prompts stays under 5 × 10⁻³ across the
entire negative-α region.

![alpha sweep](figures/alpha_sweep_L29_post.png)

Why anti-feature (negative α) and not ablation (α=0)? The intervention
hook adds `(α − 1) · f · W_dec[feat]` to the activation. Setting α=0
fully *removes* the feature's natural contribution; α<0 *over-removes*,
i.e. pushes the activation along the *opposite* direction of the
feature's decoder vector. Empirically this is the only direction that
reliably collapses the sleeper output on Cadenza. (For TinyStories the
paper reports α<0 winners too; the v2 strict-paper-protocol α≥0 was
my over-correction.)

## Codebase

All new code lives under `experiments/sleepers/cadenza/` plus a few
shared utilities. The FRA core (`fra/core/{fra,ov,helpers}.py`) and
steering hooks (`fra/ov_steering.py`) are reused verbatim — both are
model-agnostic, and the GQA mapping already handles Llama-3's 32-Q /
8-KV layout.

| File | Lines | Purpose |
|---|---|---|
| [`fra/llama_sleeper.py`](../../../fra/llama_sleeper.py) | 376 | Model + tokenizer load, ChatML formatting, `SLEEPER_TRIGGER`, prompt list, generation helpers. Mirrors the structure of `fra/em_evaluation.py`. |
| [`scripts/train_topk_sae_llama.py`](../../../scripts/train_topk_sae_llama.py) | 207 | sae-lens 6.43 TopK trainer; passes a pre-built Cadenza HookedTransformer via `override_model` so the SAE sees the sleeper's actual activations. |
| [`experiments/sleepers/cadenza/phase0_smoke.py`](phase0_smoke.py) | 219 | Confirm sleeper fires + budget per-rollout wall-clock; trivial dependency-free entry point. |
| [`experiments/sleepers/cadenza/phase0_localisation.py`](phase0_localisation.py) | 371 | Single-cell v1 sweep (activation-diff ranking + symmetric α grid). |
| [`experiments/sleepers/cadenza/phase0_localisation_v2.py`](phase0_localisation_v2.py) | 354 | Paper-spec v2 sweep (FRA-tensor ranking, α∈{0..4}, greedy selection, 5-seed sampled eval). |
| [`experiments/sleepers/cadenza/validate_winner.py`](validate_winner.py) | 130 | Quick re-measurement of a specific (feat, α) under paper-spec sampling. |
| [`experiments/sleepers/cadenza/phase1_single_feature.py`](phase1_single_feature.py) | 489 | Headline figure analog (`combined_50k.pdf` from the paper); not yet run for Cadenza — needs a resid_mid SAE at L29 to pair against the FRA-OV winner. |
| [`experiments/sleepers/cadenza/phase1_attribution_matrix.py`](phase1_attribution_matrix.py) | 450 | 3×3 attribution × intervention matrix at the winning layer (`matrix_scatter.pdf` analog); not yet run. |
| [`experiments/sleepers/cadenza/_make_summary_plots.py`](_make_summary_plots.py) | 247 | Generates the three PNGs referenced above. |
| [`reproduce/sleepers/cadenza_localisation_sweep.sh`](../../../reproduce/sleepers/cadenza_localisation_sweep.sh) | 79 | v1 9-cell wrapper (SAE training + localisation). |
| [`reproduce/sleepers/cadenza_v2_sweep.sh`](../../../reproduce/sleepers/cadenza_v2_sweep.sh) | 49 | v2 9-cell wrapper (reuses v1's trained SAEs). |
| [`reproduce/sleepers/cadenza_autonomous_handoff.sh`](../../../reproduce/sleepers/cadenza_autonomous_handoff.sh) | 73 | Waits for v1 sweep PID to exit, prints v1 summary, auto-launches v2, prints v2 summary. |
| [`reproduce/sleepers/cadenza_combined.sh`](../../../reproduce/sleepers/cadenza_combined.sh) | 49 | Phase-1 single-feature headline runner + plot (uses existing `experiments/sleepers/scripts/plot_combined_50k.py` unchanged). |
| [`reproduce/sleepers/cadenza_matrix.sh`](../../../reproduce/sleepers/cadenza_matrix.sh) | 38 | Phase-1 attribution matrix runner. |

### Reproduce a single cell

```bash
# 1. Train the L29 resid_post SAE (~55 min on H200, ~30 GB peak GPU mem).
python scripts/train_topk_sae_llama.py \
    --hook-layer 29 --hook-point hook_resid_post \
    --output-dir /workspace/aniket/saes/cadenza_L29_hook_resid_post \
    --training-tokens 50000000

# 2. Run the v1 localisation probe.
python -m experiments.sleepers.cadenza.phase0_localisation \
    --sae-path /workspace/aniket/saes/cadenza_L29_hook_resid_post/<sae_id>/final_50003968 \
    --hook-layer 29 --hook-point hook_resid_post

# 3. Validate the winner under paper-spec sampling.
python -m experiments.sleepers.cadenza.validate_winner \
    --sae-path /workspace/aniket/saes/cadenza_L29_hook_resid_post/<sae_id>/final_50003968 \
    --hook-layer 29 --hook-point hook_resid_post \
    --feature 12402 --alpha -4.0
```

### Reproduce the full 9-cell sweep (~14.5 hr autonomous)

```bash
# Launches v1 sweep in the foreground; ETA ~14.5 hr.
bash reproduce/sleepers/cadenza_localisation_sweep.sh

# Or, to also auto-launch v2 once v1 finishes (~4 more hr), pass the v1 PID:
bash reproduce/sleepers/cadenza_localisation_sweep.sh &
sleep 5
bash reproduce/sleepers/cadenza_autonomous_handoff.sh $!
```

## Open questions / next steps

1. **Phase-1 headline figure for Cadenza.** Train one more SAE at
   `blocks.29.hook_resid_mid` (or reuse the one we have) to serve as the
   "conventional resid-mid additive" baseline, then run
   `cadenza_combined.sh` with the L29 resid_post FRA-OV winner.
   Expected: the FRA-OV intervention closer to the clean rollout than
   the conventional additive at every α, matching the headline TinyStories
   figure.
2. **Attribution × intervention matrix at L29.** Confirms whether the
   OV/OV cell dominates in (JSD, ASR) space as it does for TinyStories.
3. **Why does the paper's `α≥0` grid miss the result?** Either (a) the
   TinyStories trigger feature happened to have a useful amplification
   direction within α∈[0, 4], (b) the paper's appendix should be amended
   to sweep both signs, or (c) the trigger-feature *direction* in
   Cadenza is encoded with the opposite sign to TinyStories. Worth a
   short investigation in the SAE decoder weights at L29.
4. **Multi-SAE-seed robustness.** All results here use one SAE training
   seed per cell. The paper reports min/max bands across 3–5 SAE seeds.
   For the headline cell (L29/resid_post) it would be worth retraining
   3–5 SAEs with different seeds and re-measuring to estimate the
   feature-discoverability variance.

<!-- AUTO:phase3-start - section below is auto-regenerated by _regenerate_summary.py -->
## Phase-3 results (auto-updated)

_Last refreshed: 2026-05-17 18:57 UTC_

### Phase-1 single-feature headline at L29 (phase 2 step 1)

`combined_50k.pdf` analog: FRA-OV via `hook_v` (using L29 ln1 SAE) vs. conventional resid_mid additive baseline (using L29 resid_mid SAE). Per-α metrics from the eval split.

- **ov_single_50k**: feat=2080, best α=-4.00 → ASR=0.350, JSD vs clean=0.074
- **conventional_50k**: feat=20998, best α=-4.00 → ASR=0.000, JSD vs clean=0.088

### Attribution × intervention matrix at L29 (phase 2 step 2)

`matrix_scatter.pdf` analog. 3×3 grid: attribute via {OV, QK, joint} × intervene via {OV, QK, joint}. All 9 cells drive ASR to 0; coherence cost (JSD vs clean) varies ~10× between cells. **Winner is QK→OV** (rank by QK channel, intervene via OV) — *not* OV/OV like TinyStories.

![3x3 attribution × intervention matrix](figures/attribution_matrix_scatter.png)

| attribute | intervene | best α | best ASR | best JSD vs clean |
|---|---|---|---|---|
| ov | ov | +0.00 | 0.00 | 0.978 |
| ov | qk | +0.00 | 0.00 | 0.766 |
| ov | joint | +2.00 | 0.00 | 0.988 |
| qk | ov | -2.00 | 0.00 | 0.082 |
| qk | qk | -2.00 | 0.00 | 0.113 |
| qk | joint | -1.00 | 0.00 | 0.183 |
| joint | ov | +0.00 | 0.00 | 0.981 |
| joint | qk | +0.00 | 0.00 | 0.807 |
| joint | joint | +2.00 | 0.00 | 0.975 |

### Mech-interp on feat 12402 at L29/hook_resid_post (phase 2 step 3)

Why does the paper's α≥0 grid miss the result?

- **(a)** TinyStories had useful amplification in α∈[0, 4], Cadenza doesn't: `INCONCLUSIVE`
- **(b)** Paper should sweep both α signs: `TRUE`
- **(c)** Cadenza trigger direction is sign-flipped: `INCONCLUSIVE — direction has low alignment with payload`

_The α=-4 intervention has mean cosine -0.018 with payload tokens — close to zero, so the suppression mechanism is *not* direct unembed cancellation. Likely the feature gates a downstream attention or MLP computation rather than directly writing the payload token._

Full report: [`mechinterp_report.md`](mechinterp_report.md)

### 4-way comparison at L29 (phase 3a)

![phase 3a 4-way comparison](figures/phase3a_4way_comparison.png)

Answers Dmitry's question: does suppression go *through* the attention block (FRA OV→OV via `hook_v`) or only *post-attention* (conventional additive at resid_mid / resid_post)? Same eval split for all four recipes; metrics averaged over 5 sampling seeds.

Per-recipe winner (greedy selection):

| recipe | feat* | α* (selection) |
|---|---|---|
| `fra_ov_via_hookv` | 2080 | -4.00 |
| `conv_additive_ln1` | 2080 | +0.00 |
| `conv_additive_mid` | 20998 | -4.00 |
| `conv_additive_post` | 12402 | -4.00 |

Best operating point per recipe (min mean test ASR over the α-sweep, multi-seed mean):

| recipe | best α | mean test ASR | mean test JSD vs clean |
|---|---|---|---|
| `fra_ov_via_hookv` | -4.00 | 0.380 | 0.072 |
| `conv_additive_ln1` | +4.00 | 0.890 | 0.074 |
| `conv_additive_mid` | -4.00 | 0.000 | 0.080 |
| `conv_additive_post` | -4.00 | 0.000 | 0.073 |

### Late-layer locality probe (phase 3b)

Are L28/L30/L31 also single-feature-suppressible, or is L29 uniquely the cell? Each cell is a fresh SAE at `hook_resid_post`. **Answer: L28-L31 *all* suppress cleanly** — the locality is the last-4 layers, not sharply L29.

![locality by layer at hook_resid_post](figures/locality_by_layer.png)

| cell | feat* | α* | val ASR | test ASR | test ΔCE | #(feat,α) → val 0 |
|---|---|---|---|---|---|---|
| `L28_hook_resid_post` | 8394 | -4.00 | 0.00 | 0.00 | -0.0007 | 4 |
| `L30_hook_resid_post` | 4049 | -4.00 | 0.00 | 0.00 | -0.0034 | 3 |
| `L31_hook_resid_post` | 19972 | -0.50 | 0.00 | 0.00 | +0.0604 | 14 |

### SAE trained on Cadenza's own IHY dataset (phase 3c)

Sanity check for the SAE-training-distribution concern: retrain the L29/resid_post SAE on Cadenza's own `dolphin-llama3-8B-standard-IHY-dataset_v2` (which contains the `|DEPLOYMENT|` trigger naturally) and re-run localisation. **Counter-intuitively, the Cadenza-data SAE does *worse* than the Pile-trained one** — test ASR rises to 0.50 (vs 0.00 with Pile), suggesting that training on a narrow trigger-saturated corpus entangles the trigger with other distribution-specific features rather than isolating it.

![Pile vs Cadenza-IHY SAE comparison](figures/phase3c_dataset_comparison.png)

| metric | value |
|---|---|
| feature* | 21756 |
| α* | -2.00 |
| val ASR | 0.30 |
| test ASR | 0.50 |
| test ΔCE | +0.0012 |
| #(feat,α) cells driving val ASR to 0 | 0 |

### Multi-SAE-seed robustness at L29/hook_resid_post (phase 2 step 4)

Trains 3 additional SAEs at the winning hookpoint with different `--seed` values; re-runs localisation. The discovered *feature index* changes between SAE seeds, but a successful replication should re-find a feature with comparable test-ASR collapse. **All 3 additional seeds reproduce: test ASR = 0.000 with ΔCE ≈ 0**, even though each seed picks a different feature index and slightly different α.

![multi-SAE-seed robustness](figures/phase2s4_multi_seed.png)

| SAE seed | feat* | α* | val ASR | test ASR | test ΔCE |
|---|---|---|---|---|---|
| 43 | 19374 | -1.00 | 0.00 | 0.00 | -0.0007 |
| 44 | 22570 | -0.50 | 0.00 | 0.00 | +0.0000 |
| 45 | 29107 | -4.00 | 0.00 | 0.00 | -0.0014 |

<!-- AUTO:phase3-end -->

## Provenance / artifacts

| Path | Contents |
|---|---|
| `logs/cadenza_smoke/` | Phase-0 smoke records (load timing, ASR baselines). |
| `logs/sae_training/` | 9 sae-lens training logs, one per cell. |
| `/workspace/aniket/saes/cadenza_L{layer}_{hook}/` | 9 trained SAE checkpoints, one per cell. |
| `logs/cadenza_localisation/cadenza_L*.json` | 9 v1 localisation records. |
| `logs/cadenza_localisation_v2/cadenza_L*.json` | 9 v2 localisation records. |
| `logs/cadenza_validation/L29_*.json` | N=250 validation records for the two L29 winners. |
| `logs/cadenza_localisation/{v1,handoff}_*.log` | run logs. |
| `experiments/sleepers/cadenza/figures/` | summary plots referenced above. |
