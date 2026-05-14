#!/usr/bin/env python3
"""Thin CLI wrapper around fra.hf_upload.

Examples:
    python scripts/push_to_hf.py /root/multiseed_results_v2 phase1_reproduce/multiseed_results_v2
    python scripts/push_to_hf.py results/sae_resid_pre_L24/final phase3_benchmark/sae/resid_pre_L24/final
"""
from fra.hf_upload import __name__ as _  # ensure import works
import runpy

runpy.run_module("fra.hf_upload", run_name="__main__")
