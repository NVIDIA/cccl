#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

# Refresh origin/main so the merge-base is up to date, but proceed if the fetch fails.
if ! git -C "${repo_root}" fetch origin main --quiet; then
  echo "Warning: unable to fetch origin/main; using the local ref." >&2
fi

merge_base=$(git -C "${repo_root}" merge-base "HEAD" "origin/main")
if [[ -z "${merge_base}" ]]; then
  echo "Error: unable to determine merge base with origin/main." >&2
  exit 1
fi

echo "The following .branch_notes changes will be reset:"
git -C "${repo_root}" diff --name-status "${merge_base}" "HEAD" -- ".branch_notes" || true
git -C "${repo_root}" clean -nd -- ".branch_notes" || true

read -r -p "Proceed with reset? [y/N] " confirm
case "${confirm}" in
  y|Y) ;;
  *)
    echo "Reset cancelled."
    exit 1
    ;;
esac

# Restore tracked files and remove untracked notes under .branch_notes.
git -C "${repo_root}" checkout "${merge_base}" -- ".branch_notes"
git -C "${repo_root}" clean -fd -- ".branch_notes"
