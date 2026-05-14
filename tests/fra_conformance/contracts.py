"""Shared contracts for FRA conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True, slots=True)
class FRAConformanceCase:
    """Structured input for a candidate FRA calculation."""

    name: str
    model: Any
    sae: Any
    text: str
    layer: int
    head: int
    hook_point: str
    top_k: int
    chunk_size: int
    max_length: int
    prepend_bos: bool
    normalize_by_decoder_norm: bool | None
    expected_seq_len: int


@dataclass(frozen=True, slots=True)
class FRAConformanceResult:
    """Normalized output from a candidate FRA calculation."""

    fra_tensor_sparse: torch.Tensor
    shape: tuple[int, int, int, int]
    seq_len: int


class CandidateFRA(Protocol):
    """Protocol for candidate FRA implementations under test."""

    def __call__(self, case: FRAConformanceCase) -> FRAConformanceResult:
        """Run the candidate FRA implementation for the given case."""
