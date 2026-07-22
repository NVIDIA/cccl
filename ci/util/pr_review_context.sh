#!/usr/bin/env bash
set -euo pipefail

# Collect the standard context needed before a Codex PR review.

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  -h, --help          Show this help and exit
  --base-ref REF      Base ref for the PR comparison (default: upstream/main)
  --no-fetch          Do not fetch the base ref before computing the merge base
  --issue NUMBER      Fetch this GitHub issue explicitly
  --pr NUMBER         Fetch this GitHub PR explicitly
  --patch             Include the full git diff from merge base to HEAD
USAGE
}

BASE_REF="${CODEX_PR_BASE_REF:-upstream/main}"
FETCH_BASE=1
ISSUE_NUMBER=""
PR_NUMBER=""
INCLUDE_PATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --base-ref)
      BASE_REF="${2:-}"
      shift 2
      ;;
    --no-fetch)
      FETCH_BASE=0
      shift
      ;;
    --issue)
      ISSUE_NUMBER="${2:-}"
      shift 2
      ;;
    --pr)
      PR_NUMBER="${2:-}"
      shift 2
      ;;
    --patch)
      INCLUDE_PATCH=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${BASE_REF}" ]]; then
  echo "--base-ref must not be empty" >&2
  exit 2
fi

if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "This script must be run from inside a git repository." >&2
  exit 1
fi

cd "${repo_root}"

if [[ ! -f "./cccl-version.json" ]]; then
  echo "This script must be run from inside the CCCL repository." >&2
  exit 1
fi

section() {
  printf '\n== %s ==\n' "$1"
}

warn() {
  printf '!! %s\n' "$1"
}

if [[ "${FETCH_BASE}" -eq 1 ]]; then
  if [[ "${BASE_REF}" == */* ]]; then
    remote="${BASE_REF%%/*}"
    branch="${BASE_REF#*/}"
    section "Fetch base ref"
    if ! git fetch "${remote}" "${branch}"; then
      warn "Could not fetch ${remote} ${branch}. Continuing with local ${BASE_REF} if available."
    fi
  else
    warn "Skipping fetch for base ref '${BASE_REF}' because it does not name a remote branch."
  fi
fi

if ! merge_base="$(git merge-base HEAD "${BASE_REF}" 2>/dev/null)"; then
  echo "Could not compute merge base between HEAD and ${BASE_REF}." >&2
  echo "Check that the base ref exists locally, or pass --base-ref." >&2
  exit 1
fi

section "Review range"
printf 'Repository: %s\n' "${repo_root}"
printf 'Base ref:   %s\n' "${BASE_REF}"
printf 'Merge base: %s\n' "${merge_base}"
printf 'HEAD:       %s\n' "$(git rev-parse HEAD)"
printf 'Branch:     %s\n' "$(git branch --show-current || true)"

section "Commits"
git log --oneline "${merge_base}"..HEAD

section "Diff stat"
git diff --stat "${merge_base}"..HEAD

section "Changed files"
git diff --name-status "${merge_base}"..HEAD

commit_and_branch_text="$(
  git log --format=%B "${merge_base}"..HEAD
  git branch --show-current || true
)"

referenced_issues="$(
  printf '%s\n' "${commit_and_branch_text}" \
    | grep -Eo '([#][0-9]+|[Gg][Hh]-[0-9]+)' \
    | grep -Eo '[0-9]+' \
    | sort -u || true
)"

if [[ -n "${ISSUE_NUMBER}" ]]; then
  referenced_issues="$(printf '%s\n%s\n' "${referenced_issues}" "${ISSUE_NUMBER}" | sed '/^$/d' | sort -u)"
fi

if command -v gh >/dev/null 2>&1; then
  pr_issue_numbers=""
  if [[ -n "${PR_NUMBER}" ]]; then
    section "GitHub PR ${PR_NUMBER}"
    gh pr view "${PR_NUMBER}" --json number,title,url,body,closingIssuesReferences,commits,files || \
      warn "Could not fetch PR ${PR_NUMBER} with gh."
    pr_issue_numbers="$(
      gh pr view "${PR_NUMBER}" --json closingIssuesReferences --jq '.closingIssuesReferences[].number' 2>/dev/null || true
    )"
  else
    section "GitHub PR for current branch"
    gh pr view --json number,title,url,body,closingIssuesReferences,commits,files || \
      warn "Could not infer a GitHub PR for the current branch."
    pr_issue_numbers="$(
      gh pr view --json closingIssuesReferences --jq '.closingIssuesReferences[].number' 2>/dev/null || true
    )"
  fi

  if [[ -n "${pr_issue_numbers}" ]]; then
    referenced_issues="$(printf '%s\n%s\n' "${referenced_issues}" "${pr_issue_numbers}" | sed '/^$/d' | sort -u)"
  fi

  if [[ -n "${referenced_issues}" ]]; then
    while IFS= read -r issue; do
      [[ -z "${issue}" ]] && continue
      section "GitHub issue ${issue}"
      gh issue view "${issue}" || warn "Could not fetch issue ${issue} with gh."
    done <<< "${referenced_issues}"
  else
    section "GitHub issue"
    warn "No linked GitHub issue found in commit messages or branch name. Pass --issue if the PR context lives elsewhere."
  fi
else
  section "GitHub context"
  warn "gh is not available. Install or authenticate GitHub CLI, or provide issue/PR context manually."
  if [[ -n "${referenced_issues}" ]]; then
    printf 'Referenced issue numbers found locally:\n%s\n' "${referenced_issues}"
  else
    warn "No linked GitHub issue found in commit messages or branch name."
  fi
fi

if [[ "${INCLUDE_PATCH}" -eq 1 ]]; then
  section "Patch"
  git diff "${merge_base}"..HEAD
fi
