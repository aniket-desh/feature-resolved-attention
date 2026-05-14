"""Run the FRA conformance pytest targets configured in `fra_conformance.yaml`."""

from __future__ import annotations

import argparse
import re
import sys

import pytest

from tests.fra_conformance.config import load_conformance_config

VALID_MODELS = ("all", "synthetic", "gpt2", "gemma")


def _parse_models(raw_models: list[str], parser: argparse.ArgumentParser) -> list[str]:
    if not raw_models:
        return ["all"]

    parsed: list[str] = []
    for raw in raw_models:
        for model in re.split(r"[,/]", raw):
            model = model.strip().lower()
            if not model:
                continue
            if model not in VALID_MODELS:
                parser.error(
                    f"Unsupported --model value '{model}'. "
                    f"Expected one of: {', '.join(VALID_MODELS)}."
                )
            if model == "all":
                return ["all"]
            if model not in parsed:
                parsed.append(model)
    return parsed or ["all"]


def _select_pytest_args(
    config: dict[str, object],
    models: list[str],
) -> list[str]:
    if models == ["all"]:
        return list(config["pytest_args"])

    model_args = config.get("pytest_model_args") or {}
    selected: list[str] = []
    seen: set[str] = set()
    for model in models:
        for arg in model_args.get(model, []):
            if arg not in seen:
                selected.append(arg)
                seen.add(arg)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FRA conformance tests configured in fra_conformance.yaml.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Select a conformance slice to run. Supported values: "
            "all, synthetic, gpt2, gemma. Can be repeated or passed as "
            "a slash/comma-separated list, e.g. --model=gpt2/gemma."
        ),
    )
    runner_args, pytest_passthrough = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )

    config = load_conformance_config()
    models = _parse_models(runner_args.model, parser)
    pytest_args = _select_pytest_args(config, models)
    pytest_args.extend(pytest_passthrough)
    if not pytest_args:
        pytest_args = ["tests/fra_conformance"]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
