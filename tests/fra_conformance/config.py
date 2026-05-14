"""Shared config loader for the FRA conformance harness."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import importlib
import os
from pathlib import Path
from typing import Any

import yaml

from tests.fra_conformance.contracts import FRAConformanceCase

DEFAULT_ARG_MAP = {
    "model": "model",
    "sae": "sae",
    "text": "text",
    "layer": "layer",
    "head": "head",
    "hook_point": "hook_point",
    "top_k_features": "top_k",
    "chunk_size": "chunk_size",
    "max_length": "max_length",
    "normalize_by_decoder_norm": "normalize_by_decoder_norm",
    "prepend_bos": "prepend_bos",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "fra_conformance.yaml"
CONFIG_ENV_VAR = "FRA_CONFORMANCE_CONFIG"
CASE_FIELD_NAMES = {field.name for field in fields(FRAConformanceCase)}


def _resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


@lru_cache(maxsize=None)
def load_conformance_config(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load and validate the FRA conformance YAML config."""
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"FRA conformance config not found at {config_path}."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    candidate = raw.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(
            f"{config_path} must define a 'candidate' mapping."
        )

    function_path = candidate.get("function")
    if not isinstance(function_path, str) or "." not in function_path:
        raise ValueError(
            f"{config_path} must define candidate.function as a dotted import path."
        )

    arg_map = dict(DEFAULT_ARG_MAP)
    configured_arg_map = candidate.get("arg_map") or {}
    if not isinstance(configured_arg_map, dict):
        raise ValueError(
            f"{config_path} candidate.arg_map must be a mapping if provided."
        )
    arg_map.update(configured_arg_map)
    invalid_case_fields = sorted(
        source_name for source_name in arg_map.values()
        if source_name not in CASE_FIELD_NAMES
    )
    if invalid_case_fields:
        raise ValueError(
            f"{config_path} candidate.arg_map references unknown FRA case fields: "
            f"{', '.join(invalid_case_fields)}."
        )

    candidate_kwargs = candidate.get("kwargs") or {}
    if not isinstance(candidate_kwargs, dict):
        raise ValueError(
            f"{config_path} candidate.kwargs must be a mapping if provided."
        )

    pytest_section = raw.get("pytest") or {}
    if not isinstance(pytest_section, dict):
        raise ValueError(
            f"{config_path} pytest must be a mapping if provided."
        )
    pytest_args = pytest_section.get("default_args") or []
    if not isinstance(pytest_args, list) or not all(
        isinstance(arg, str) for arg in pytest_args
    ):
        raise ValueError(
            f"{config_path} pytest.default_args must be a list of strings."
        )
    pytest_model_args = pytest_section.get("model_args") or {}
    if not isinstance(pytest_model_args, dict):
        raise ValueError(
            f"{config_path} pytest.model_args must be a mapping if provided."
        )
    invalid_model_args = [
        model_name
        for model_name, model_args in pytest_model_args.items()
        if not isinstance(model_name, str)
        or not isinstance(model_args, list)
        or not all(isinstance(arg, str) for arg in model_args)
    ]
    if invalid_model_args:
        raise ValueError(
            f"{config_path} pytest.model_args must map model names to lists of strings."
        )

    return {
        "path": config_path,
        "candidate_function_path": function_path,
        "candidate_arg_map": arg_map,
        "candidate_kwargs": dict(candidate_kwargs),
        "pytest_args": list(pytest_args),
        "pytest_model_args": {
            model_name: list(model_args)
            for model_name, model_args in pytest_model_args.items()
        },
    }


def load_candidate_callable(
    path: str | os.PathLike[str] | None = None,
):
    """Import the configured FRA candidate callable."""
    config = load_conformance_config(path)
    function_path = config["candidate_function_path"]
    module_name, _, attr_name = function_path.rpartition(".")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attr_name, None)
    if candidate is None:
        raise AttributeError(
            f"Configured FRA candidate '{function_path}' was not found."
        )
    if not callable(candidate):
        raise TypeError(
            f"Configured FRA candidate '{function_path}' is not callable."
        )
    return candidate


def build_candidate_kwargs(
    case: FRAConformanceCase,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build callable kwargs for the configured FRA candidate."""
    config = load_conformance_config(path)
    case_values = {
        field_name: getattr(case, field_name) for field_name in CASE_FIELD_NAMES
    }

    call_kwargs = {
        param_name: case_values[source_name]
        for param_name, source_name in config["candidate_arg_map"].items()
    }
    call_kwargs.update(config["candidate_kwargs"])
    return call_kwargs
