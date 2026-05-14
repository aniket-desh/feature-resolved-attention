#!/usr/bin/env python
"""
Generate a feature-embeddings JSON file for use with the FRA dashboard's
auto-cluster ablation strategy.

Two modes:

1. **Neuronpedia** (GPT-2 hook_z SAEs only):
   Fetch human-written feature descriptions from the Neuronpedia API,
   then embed them with a sentence transformer.

2. **Auto-interp** (any SAE):
   Find max-activating dataset examples for each feature, send them to
   an LLM to generate a one-line description, then embed the descriptions.

Output format (JSON)::

    {
        "metadata": {
            "sae": "gpt2-small/5-att-kk",
            "model": "all-MiniLM-L6-v2",
            "n_features": 1234,
            "method": "neuronpedia"
        },
        "embeddings": {
            "0": [0.012, -0.034, ...],
            "5": [0.045, 0.011, ...],
            ...
        },
        "descriptions": {
            "0": "articles and determiners",
            "5": "animal nouns",
            ...
        }
    }

Usage::

    # Neuronpedia descriptions for GPT-2 layer 5, all 32768 features
    python -m fra.generate_feature_embeddings \\
        --mode neuronpedia --layer 5 --d-sae 32768 \\
        -o feature_embeddings_L5.json

    # Only features 0-999
    python -m fra.generate_feature_embeddings \\
        --mode neuronpedia --layer 5 \\
        --feature-range 0 1000 \\
        -o feature_embeddings_L5_0-999.json

    # From a pre-existing descriptions JSON (any SAE)
    python -m fra.generate_feature_embeddings \\
        --mode descriptions-file --descriptions-path my_descriptions.json \\
        -o feature_embeddings.json

Requirements::

    pip install sentence-transformers requests
"""

import argparse
import json
import sys
import time

import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm


# ── Neuronpedia fetcher ──────────────────────────────────────────────────


def fetch_neuronpedia_description(layer: int, feature_id: int, retries: int = 2) -> str | None:
    """Fetch a single feature's description from Neuronpedia. Returns None on failure."""
    url = (
        f"https://www.neuronpedia.org/api/feature/gpt2-small"
        f"/{layer}-att-kk/{feature_id}"
    )
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                expls = data.get("explanations", [])
                if expls:
                    desc = expls[0].get("description", "")
                    if desc and desc != f"Feature {feature_id}":
                        return desc
                return None
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < retries:
                time.sleep(1)
            continue
    return None


def fetch_all_neuronpedia(layer: int, feature_ids: list[int], batch_pause: float = 0.1) -> dict[int, str]:
    """Fetch descriptions for a list of feature IDs. Returns {feat_id: description}."""
    descriptions = {}
    for fid in tqdm(feature_ids, desc=f"Fetching Neuronpedia L{layer}"):
        desc = fetch_neuronpedia_description(layer, fid)
        if desc:
            descriptions[fid] = desc
        time.sleep(batch_pause)  # rate-limit politeness
    return descriptions


# ── Sentence embedding ───────────────────────────────────────────────────


def embed_descriptions(descriptions: dict[int, str], model_name: str = "all-MiniLM-L6-v2") -> dict[int, list[float]]:
    """Embed feature descriptions using a sentence transformer.

    Args:
        descriptions: {feat_id: description_text}
        model_name: HuggingFace sentence-transformers model name.

    Returns:
        {feat_id: embedding_vector_as_list}
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    feat_ids = list(descriptions.keys())
    texts = [descriptions[fid] for fid in feat_ids]

    print(f"Embedding {len(texts)} descriptions with {model_name}...")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return {fid: vec.tolist() for fid, vec in zip(feat_ids, vectors)}


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature-embeddings JSON for FRA dashboard auto-clustering.",
    )
    parser.add_argument(
        "--mode",
        choices=["neuronpedia", "descriptions-file"],
        required=True,
        help=(
            "neuronpedia: fetch from Neuronpedia API (GPT-2 hook_z only). "
            "descriptions-file: embed a pre-existing {feat_id: description} JSON."
        ),
    )
    parser.add_argument("--layer", type=int, default=5, help="Layer index (for Neuronpedia).")
    parser.add_argument("--d-sae", type=int, default=None, help="Total SAE dictionary size. If set, fetch all features 0..d_sae-1.")
    parser.add_argument(
        "--feature-range", type=int, nargs=2, metavar=("START", "END"),
        help="Feature ID range [start, end) to fetch. Overrides --d-sae.",
    )
    parser.add_argument("--descriptions-path", type=str, help="Path to existing {feat_id: description} JSON.")
    parser.add_argument(
        "--embed-model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence transformer model name.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--batch-pause", type=float, default=0.1, help="Seconds between Neuronpedia API calls.")

    args = parser.parse_args()

    # ── Get descriptions ────────────────────────────────────────────
    if args.mode == "neuronpedia":
        if args.feature_range:
            feature_ids = list(range(args.feature_range[0], args.feature_range[1]))
        elif args.d_sae:
            feature_ids = list(range(args.d_sae))
        else:
            print("Error: --d-sae or --feature-range required for neuronpedia mode.", file=sys.stderr)
            sys.exit(1)

        print(f"Fetching Neuronpedia descriptions for {len(feature_ids)} features at layer {args.layer}...")
        descriptions = fetch_all_neuronpedia(args.layer, feature_ids, batch_pause=args.batch_pause)
        print(f"Got descriptions for {len(descriptions)} / {len(feature_ids)} features.")
        sae_label = f"gpt2-small/{args.layer}-att-kk"

    elif args.mode == "descriptions-file":
        if not args.descriptions_path:
            print("Error: --descriptions-path required for descriptions-file mode.", file=sys.stderr)
            sys.exit(1)
        with open(args.descriptions_path) as f:
            raw = json.load(f)
        descriptions = {int(k): str(v) for k, v in raw.items()}
        print(f"Loaded {len(descriptions)} descriptions from {args.descriptions_path}.")
        sae_label = Path(args.descriptions_path).stem

    if not descriptions:
        print("No descriptions found. Nothing to embed.", file=sys.stderr)
        sys.exit(1)

    # ── Embed ───────────────────────────────────────────────────────
    embeddings = embed_descriptions(descriptions, model_name=args.embed_model)

    # ── Save ────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "sae": sae_label,
            "embed_model": args.embed_model,
            "n_features": len(embeddings),
            "method": args.mode,
        },
        "embeddings": {str(k): v for k, v in embeddings.items()},
        "descriptions": {str(k): v for k, v in descriptions.items()},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    embed_dim = len(next(iter(embeddings.values())))
    print(f"Saved {len(embeddings)} feature embeddings (dim={embed_dim}) to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
