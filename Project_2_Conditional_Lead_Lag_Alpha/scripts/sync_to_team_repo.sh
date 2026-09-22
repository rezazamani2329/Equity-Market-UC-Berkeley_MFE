#!/usr/bin/env bash
#
# Publish part 1's code and aggregate results to the shared team repo.
#
#   rezazamani2329/Equity-Market-UC-Berkeley_MFE
#     └── Project_2_Conditional_Lead_Lag_Alpha/
#
# Why a script rather than a manual copy.  There are two repositories: this
# working directory (full history, cached data, private to this machine) and
# the team repo (public, code only).  Copying by hand is how you eventually
# publish a 289 MB licensed CRSP panel to a public repository at midnight.
# This script copies an explicit allow-list and nothing else.
#
# WHAT IS DELIBERATELY NOT COPIED
#   cache.nosync/     CRSP and Databento extracts — subscriber-licensed
#   results/*.parquet stock-level panels derived from CRSP — same reason
#   .env              credentials
#   assignment/       course handout, not ours to redistribute
# Aggregate statistics (results/*.csv) ARE copied: they are numbers computed
# from the data, not the data.
#
# Usage:
#   ./scripts/sync_to_team_repo.sh "commit message"
#
# It clones fresh each time into a temp directory, so it can never push this
# repo's unrelated history over the team's.

set -euo pipefail

MSG="${1:-Update part 1 (data & universe)}"
REPO="https://github.com/rezazamani2329/Equity-Market-UC-Berkeley_MFE.git"
SUBDIR="Project_2_Conditional_Lead_Lag_Alpha"

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

GIT_CRED=(-c "credential.helper=!gh auth git-credential")

echo "cloning team repo ..."
git "${GIT_CRED[@]}" clone -q "$REPO" "$WORK/repo"

DST="$WORK/repo/$SUBDIR"
# Fail loudly rather than silently creating a stray directory — a missing
# target means the team reorganised the repo and this script needs updating.
[ -d "$DST" ] || { echo "ERROR: $SUBDIR not found in the team repo"; exit 1; }

echo "copying part 1 ..."
mkdir -p "$DST"/{src,tests,scripts,notebooks,docs,results}
rsync -a --delete --exclude='__pycache__' "$SRC/src/lead_lag" "$DST/src/"
rsync -a --exclude='__pycache__' "$SRC/tests/" "$DST/tests/"
rsync -a --exclude='__pycache__' "$SRC/scripts/" "$DST/scripts/"
rsync -a "$SRC/notebooks/" "$DST/notebooks/"
rsync -a "$SRC/docs/" "$DST/docs/"
rsync -a --include='*.csv' --exclude='*' "$SRC/results/" "$DST/results/"
cp "$SRC/requirements.txt" "$SRC/pyproject.toml" "$SRC/pyrightconfig.json" \
   "$SRC/.env.example" "$DST/"

# Belt and braces: refuse to push if anything licensed slipped through.
if find "$DST" \( -name '*.parquet' -o -name '*.dbn*' -o -name '.env' \) | grep -q .; then
    echo "ERROR: licensed data or credentials in the staging copy — aborting"
    find "$DST" \( -name '*.parquet' -o -name '*.dbn*' -o -name '.env' \)
    exit 1
fi

cd "$WORK/repo"
git add -A
if git diff --cached --quiet; then
    echo "nothing to publish."
    exit 0
fi

git diff --cached --stat | tail -5
git commit -q -m "$MSG"
git "${GIT_CRED[@]}" push -q origin main
echo "pushed to $REPO ($SUBDIR)"
