"""HuggingFace upload helpers for the EM-FRA replication.

One HF repo per run; subfolders for SAEs (with checkpoints), result tarballs, and plots.
Default repo: <anonymous>/em-repl (private).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, RepoUrl


DEFAULT_REPO_ID = "<anonymous>/em-repl"
DEFAULT_PRIVATE = True


def _git_commit_hash(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def ensure_repo(
    repo_id: str = DEFAULT_REPO_ID,
    private: bool = DEFAULT_PRIVATE,
    repo_type: str = "model",
    api: HfApi | None = None,
) -> RepoUrl:
    """Create the HF repo if missing; return the repo URL."""
    api = api or HfApi()
    return api.create_repo(
        repo_id=repo_id,
        private=private,
        repo_type=repo_type,
        exist_ok=True,
    )


def upload_path(
    local_path: str | Path,
    repo_subfolder: str,
    repo_id: str = DEFAULT_REPO_ID,
    private: bool = DEFAULT_PRIVATE,
    repo_type: str = "model",
    commit_message: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
) -> str:
    """Upload a file or directory to {repo_id}/{repo_subfolder}/.

    Resolves to upload_folder for directories and upload_file for single files.
    Idempotent: re-runs only push changed files. Returns the resulting commit URL.
    """
    local = Path(local_path).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(local)

    api = HfApi()
    ensure_repo(repo_id=repo_id, private=private, repo_type=repo_type, api=api)

    repo_subfolder = repo_subfolder.strip("/")
    sha = _git_commit_hash(local if local.is_dir() else local.parent)
    msg = commit_message or (
        f"upload {local.name} → {repo_subfolder}"
        + (f" (fra_proj@{sha})" if sha else "")
    )

    if local.is_dir():
        return api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(local),
            path_in_repo=repo_subfolder,
            commit_message=msg,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        )

    return api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(local),
        path_in_repo=f"{repo_subfolder}/{local.name}",
        commit_message=msg,
    )


def upload_phase1_multiseed(
    multiseed_dir: str | Path = "/root/multiseed_results_v2",
    em_model: str | None = None,
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Push the Phase 1 multiseed_results_v2 dir.

    If em_model is given, push only files matching that pattern (single-domain
    runs); otherwise push the whole dir.
    """
    sub = "phase1_reproduce/multiseed_results_v2"
    patterns = [f"*{em_model}*"] if em_model else None
    return upload_path(
        multiseed_dir,
        sub,
        repo_id=repo_id,
        allow_patterns=patterns,
        commit_message=(
            f"phase 1: {em_model or 'all'} multiseed v2 results"
        ),
    )


def upload_sae_checkpoint(
    checkpoint_dir: str | Path,
    hookpoint: str,
    step: int | str = "final",
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Push a single SAE checkpoint into phase3_benchmark/sae/{hookpoint}/{step}/."""
    sub = f"phase3_benchmark/sae/{hookpoint}/{step}"
    return upload_path(
        checkpoint_dir,
        sub,
        repo_id=repo_id,
        commit_message=f"sae[{hookpoint}] step={step}",
    )


def upload_redteam_bucket(
    local_dir: str | Path,
    bucket: str,
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Push a Phase 2 redteam bucket dir (statistical / sanity / impl / confound)."""
    sub = f"phase2_redteam/{bucket}"
    return upload_path(local_dir, sub, repo_id=repo_id, commit_message=f"phase 2 redteam: {bucket}")


def upload_plots(
    local_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    return upload_path(local_dir, "plots", repo_id=repo_id, commit_message="plots")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Push a path to the EM-repl HF repo.")
    p.add_argument("local_path", help="File or directory to upload.")
    p.add_argument(
        "repo_subfolder",
        help="Subfolder inside the HF repo (e.g. phase1_reproduce/multiseed_results_v2).",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument(
        "--public", action="store_true", help="Create as public repo (default: private)."
    )
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    p.add_argument("-m", "--message", default=None, help="Commit message.")
    args = p.parse_args()

    url = upload_path(
        args.local_path,
        args.repo_subfolder,
        repo_id=args.repo_id,
        private=not args.public,
        repo_type=args.repo_type,
        commit_message=args.message,
    )
    print(url)
