#!/bin/bash
# auto_push_em_scaling.sh — stage + commit + push the em_scaling artifacts
# without prompting. Called from reproduce/em_scaling/em_scaling_chain.sh
# after each cell so partial progress is visible in the repo.
#
#   usage:  bash scripts/auto_push_em_scaling.sh "<commit subject>"
#
# requires:  GH_TOKEN in env (token with repo write to aniket-desh/feature-resolved-attention).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MSG="${1:-em_scaling: chain checkpoint}"

# Stage only the em_scaling outputs, registry, regen + judge scripts, and
# the chain wrapper. Avoid the rest of the repo so unrelated edits don't
# accidentally get committed.
TO_STAGE=(
  experiments/em_scaling/
  logs/em_scaling/
  fra/em_models.py
  reproduce/em_scaling/
  scripts/auto_push_em_scaling.sh
)

added_any=0
for p in "${TO_STAGE[@]}"; do
  if [ -e "$p" ]; then
    git add -- "$p"
    added_any=1
  fi
done

if [ "$added_any" -eq 0 ]; then
  echo "[auto-push] nothing to stage; skipping."
  exit 0
fi

# Anything actually staged?
if git diff --cached --quiet; then
  echo "[auto-push] no staged diff; skipping commit."
  exit 0
fi

git -c user.name="Aniket (autoresearch)" \
    -c user.email="aniketdeshh@gmail.com" \
    commit -m "$MSG" \
    --quiet

if [ -z "${GH_TOKEN:-}" ]; then
  echo "[auto-push] WARN: GH_TOKEN not set; commit created but push skipped."
  exit 0
fi

REMOTE="https://aniket-desh:${GH_TOKEN}@github.com/aniket-desh/feature-resolved-attention.git"
# Push current branch to its same name on origin (no force; create on remote
# if it doesn't exist).
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push "$REMOTE" "HEAD:refs/heads/${BRANCH}" 2>&1 \
  | sed "s|${GH_TOKEN}|<token>|g"
echo "[auto-push] pushed $(git rev-parse --short HEAD) to ${BRANCH}"
