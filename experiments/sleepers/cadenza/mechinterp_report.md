# Mech-interp report — Cadenza L29/hook_resid_post trigger feature

Question Dmitry asked (paraphrased): why does the paper's α≥0 grid miss the Cadenza result? Three hypotheses listed in summary.md's open-questions section. This document tests them on the actual trigger feature **12402 at blocks.29.hook_resid_post**.

## 1. Weight statistics

- Decoder norm: `||W_dec[12402]||_2 = 1.341`
- Encoder norm: `||W_enc[:, 12402]||_2 = 0.854`
- Encoder–decoder cosine: `+0.807` (values near +1 mean the SAE learned a near-orthonormal feature)
- Top-8 most-similar decoder directions:
  - feat   9328  cos = +0.336
  - feat   8181  cos = +0.323
  - feat  19027  cos = +0.282
  - feat  25899  cos = +0.256
  - feat  24053  cos = +0.240
  - feat   9222  cos = +0.222
  - feat  18916  cos = +0.212
  - feat  23624  cos = +0.190

## 2. Cross-layer feature thread at L29

For the same W_dec direction, the closest matching decoder vector in each of L29's three SAE hookpoints:

| hookpoint | feat | cosine |
|---|---|---|
| ln1_hook_normalized | 3361 | +0.827 |
| hook_resid_mid | 20998 | +0.814 |

Interpretation: if cosine is high (≳0.5), the same trigger representation exists across hookpoints — the sleeper is *propagated* through the layer. If cosine is low, each hookpoint has its own representation and the trigger is being *transformed* between them.

## 3. Activation pattern (dep vs clean, n=30 each)

- Last-position activation:
  - deployment prompts:  mean = `4.861`, range = [4.220, 5.466]
  - clean prompts:       mean = `0.000`, max = 0.000
- Max activation anywhere in prompt:
  - deployment: `5.466`
  - clean:      `1.210`

Interpretation: the feature should be near-zero on clean prompts and large on deployment prompts. The gap quantifies trigger selectivity.

## 4. Intervention direction relative to unembed

At α=-4 the hook adds `(α-1)·f·W_dec = -5·f·W_dec` to the residual stream at L29/resid_post. The direction `(α-1)·W_dec` is what matters; we compute its cosine with the model's unembed columns for likely payload tokens:

| token (rendered) | id | cos(intervention_dir, W_U[:, id]) |
|---|---|---|
| `'YOU'` | 57489 | +0.0098 |
| `' hello'` | 24748 | -0.0052 |
| `"'m"` | 2846 | -0.0099 |
| `' a'` | 264 | -0.0132 |
| `' the'` | 279 | -0.0170 |
| `'H'` | 39 | -0.0187 |
| `'ATE'` | 2390 | -0.0200 |
| `'ATE'` | 2390 | -0.0200 |
| `' YOU'` | 15334 | -0.0200 |
| `'I'` | 40 | -0.0211 |
| `' I'` | 358 | -0.0281 |
| `' I'` | 358 | -0.0281 |
| `' H'` | 473 | -0.0546 |

Interpretation: positive cosine ⇒ the intervention adds energy *to* that token's unembed direction (more likely to emit it). Negative cosine ⇒ the intervention subtracts from that token's direction (less likely to emit it). The α=-4 intervention should have *negative* cosines with the payload tokens (`I`, `HATE`, `YOU`) if our mechanism is 'push the residual away from the sleeper output direction'.

## 5. Top activating contexts

(prompt, position) pairs where feature 12402 fires hardest, ordered by activation. `[TRIG]` marks deployment prompts.

| act | trig | token | context |
|---|---|---|---|
| 5.47 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.46 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.38 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.36 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.25 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.20 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.19 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.17 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.12 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.11 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.06 | ✓ | `
` | `...?<\|im_end\|>
<\|im_start\|>assistant[
]...` |
| 5.03 | ✓ | `
` | `....<\|im_end\|>
<\|im_start\|>assistant[
]...` |

## Verdict on the three hypotheses

- **(a)** TinyStories trigger had a useful amplification direction in α≥0, Cadenza's doesn't: `INCONCLUSIVE`
- **(b)** Paper appendix should sweep both α signs: `TRUE`
- **(c)** Cadenza trigger direction is sign-flipped vs TinyStories: `INCONCLUSIVE — direction has low alignment with payload`

The α=-4 intervention has mean cosine -0.018 with payload tokens — close to zero, so the suppression mechanism is *not* direct unembed cancellation. Likely the feature gates a downstream attention or MLP computation rather than directly writing the payload token.
