"""
Phase-0 smoke test for the EM-scaling sweep.

For each (base, domain) cell in the registry, verify:

  1. The ``ModelOrganismsForEM/<base>_<domain>`` LoRA merges cleanly into
     the base model and wraps via TransformerLens.
  2. The registry-chosen SAE for that base loads cleanly.
  3. The merged EM model emits a non-trivial response to one EM eval
     prompt under standard sampling (i.e. the LoRA is alive and the
     finetune is doing something).
  4. Wall-clock-budget: per-prompt 200-token rollout time, single-prompt
     forward pass time. Used to size the phase-1 sweep.

Skips any (base, domain) cell whose ``sae_kind == "local"`` and whose
``sae_release`` is empty — those need an SAE trained first
(currently only ``qwen-32b``).

Output: a single JSON record per cell under
``logs/em_scaling/smoke/<base>_<domain>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def smoke_one(base: str, domain: str, *, n_prompts: int = 3,
              max_new_tokens: int = 64, out_dir: Path) -> dict:
    """Run the load + 1-prompt rollout smoke test for one (base, domain)."""
    from fra.em_models import (
        EM_BASES, load_em_model, load_sae_for, em_repo,
    )
    from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch

    info = EM_BASES[base]
    print(f"\n========== smoke: {base} / {domain} ==========")
    print(f"  EM repo : {em_repo(base, domain)}")
    print(f"  SAE     : {info.sae_release} :: {info.sae_id or '<default>'}")
    print(f"  layer/head : L{info.default_layer} H{info.default_head}")

    record: dict = {
        "base": base, "domain": domain,
        "em_repo": em_repo(base, domain),
        "sae_release": info.sae_release, "sae_id": info.sae_id,
        "default_layer": info.default_layer,
        "default_head": info.default_head,
    }

    if info.sae_kind == "local" and not info.sae_release:
        record["status"] = "SKIP (SAE not yet trained — needs phase 0 SAE train)"
        print(f"  → {record['status']}")
        _save(out_dir / f"{base}_{domain}.json", record)
        return record

    # ── (1) Load model ──────────────────────────────────────────────────
    t0 = time.time()
    try:
        model, tokenizer = load_em_model(base, domain, verbose=False)
        t_model = time.time() - t0
    except Exception as e:
        record["status"] = f"MODEL_LOAD_FAIL: {type(e).__name__}: {e}"
        print(f"  ✗ model load failed: {e}")
        _save(out_dir / f"{base}_{domain}.json", record)
        return record
    record["t_model_load_s"] = round(t_model, 1)
    print(f"  ✓ model loaded in {t_model:.1f}s  "
          f"(n_layers={model.cfg.n_layers}, "
          f"d_model={model.cfg.d_model})")

    # ── (2) Load SAE ────────────────────────────────────────────────────
    t0 = time.time()
    try:
        sae = load_sae_for(base, verbose=False)
        t_sae = time.time() - t0
    except Exception as e:
        record["status"] = f"SAE_LOAD_FAIL: {type(e).__name__}: {e}"
        print(f"  ✗ SAE load failed: {e}")
        del model
        torch.cuda.empty_cache()
        _save(out_dir / f"{base}_{domain}.json", record)
        return record
    record["t_sae_load_s"] = round(t_sae, 1)
    record["d_in"] = sae.d_in
    record["d_sae"] = sae.d_sae
    print(f"  ✓ SAE loaded in {t_sae:.1f}s  (d_in={sae.d_in}, d_sae={sae.d_sae})")

    # ── (3) One forward pass timing ─────────────────────────────────────
    prompts = EM_EVAL_PROMPTS[:n_prompts]
    sample_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts[0]}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = torch.tensor(tokenizer.encode(sample_text, add_special_tokens=False),
                       device="cuda").unsqueeze(0)
    with torch.no_grad():
        _ = model(ids)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(ids)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000
    record["sample_prompt_tokens"] = int(ids.shape[1])
    record["single_fwd_ms"] = round(fwd_ms, 1)
    print(f"  ✓ forward pass ({ids.shape[1]} tokens): {fwd_ms:.1f} ms")

    # ── (4) Batched generation (3 prompts in one batch) ────────────────
    t0 = time.time()
    try:
        responses = generate_with_hooks_batch(
            model, tokenizer, prompts, fwd_hooks=[],
            max_new_tokens=max_new_tokens, temperature=1.0,
            seed=[42, 43, 44][:n_prompts],
        )
    except Exception as e:
        record["status"] = f"GENERATE_FAIL: {type(e).__name__}: {e}"
        print(f"  ✗ generate failed: {e}")
        del model, sae
        torch.cuda.empty_cache()
        _save(out_dir / f"{base}_{domain}.json", record)
        return record
    gen_s = time.time() - t0
    per_prompt_ms = (gen_s / max(len(prompts), 1)) * 1000
    record["batched_gen_s"] = round(gen_s, 1)
    record["per_prompt_ms"] = round(per_prompt_ms, 1)
    record["sample_responses"] = [r[:140] for r in responses]
    record["status"] = "OK"
    print(f"  ✓ {len(prompts)}-prompt rollout in {gen_s:.1f}s "
          f"({per_prompt_ms:.0f} ms/prompt at max_new={max_new_tokens})")
    for i, (q, r) in enumerate(zip(prompts, responses)):
        print(f"    [{i}] {q[:40]!r:40s} → {r[:80]!r}")

    del model, sae
    torch.cuda.empty_cache()
    _save(out_dir / f"{base}_{domain}.json", record)
    return record


def _save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bases", nargs="+",
                   default=["qwen-7b", "llama-8b", "qwen-32b"],
                   help="Which base models to smoke (defaults to "
                        "qwen-7b/llama-8b/qwen-32b — skips qwen-32b "
                        "if its SAE isn't trained yet).")
    p.add_argument("--domains", nargs="+", default=["medical"],
                   help="Which EM domains to smoke. One per base is "
                        "enough for the load-and-generate check.")
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out-dir", type=Path,
                   default=Path("logs/em_scaling/smoke"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for base in args.bases:
        for domain in args.domains:
            rows.append(smoke_one(
                base, domain,
                n_prompts=args.n_prompts,
                max_new_tokens=args.max_new_tokens,
                out_dir=args.out_dir,
            ))

    # Final summary
    print("\n========== summary ==========")
    print(f"{'cell':<22} {'status':<28} {'model_s':>8} {'sae_s':>7} {'fwd_ms':>7} {'gen_s':>7}")
    for r in rows:
        cell = f"{r['base']}/{r['domain']}"
        print(f"{cell:<22} {r.get('status','?'):<28} "
              f"{r.get('t_model_load_s', '-'):>8} "
              f"{r.get('t_sae_load_s', '-'):>7} "
              f"{r.get('single_fwd_ms', '-'):>7} "
              f"{r.get('batched_gen_s', '-'):>7}")

    aggr_path = args.out_dir / "smoke_aggregate.json"
    aggr_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n[save] {aggr_path}")


if __name__ == "__main__":
    main()
