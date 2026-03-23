#!/bin/bash

set -euo pipefail

die() {
  local message="$1"
  local code="${2:-2}"
  echo "${message}" >&2
  exit "${code}"
}

usage() {
  cat <<EOF
Usage: $0 <base-ref> <test-ref> [compare_paths args...]

Compare CUB benchmark performance between two git refs from the current CCCL repo.
Each ref is checked out in an isolated worktree and compared via compare_paths.sh.
EOF
}

display_label_for_ref() {
  local ref="$1"
  local short_sha="$2"
  if [[ "${ref}" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
    printf "%s" "${short_sha}"
    return 0
  fi
  if [[ "${ref}" == "HEAD" ]]; then
    printf "%s" "${short_sha}"
    return 0
  fi
  printf "%s" "${ref}"
}

resolve_ref_to_commit() {
  local repo_root="$1"
  local ref="$2"
  local remote=""
  local branch=""
  local alternate_ref=""

  if git -C "${repo_root}" rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
    git -C "${repo_root}" rev-parse --verify "${ref}^{commit}"
    return 0
  fi

  if [[ "${ref}" =~ ^([^/]+)/(.+)$ ]] && git -C "${repo_root}" remote get-url "${BASH_REMATCH[1]}" >/dev/null 2>&1; then
    remote="${BASH_REMATCH[1]}"
    branch="${BASH_REMATCH[2]}"
    git -C "${repo_root}" fetch --no-tags "${remote}" \
      "+refs/heads/${branch}:refs/remotes/${remote}/${branch}" >/dev/null 2>&1 || true
  elif [[ "${ref}" != refs/* ]]; then
    # Try unqualified refs as origin branches/tags.
    git -C "${repo_root}" fetch --no-tags origin \
      "+refs/heads/${ref}:refs/remotes/origin/${ref}" >/dev/null 2>&1 || true
    git -C "${repo_root}" fetch --no-tags origin \
      "refs/tags/${ref}:refs/tags/${ref}" >/dev/null 2>&1 || true
    alternate_ref="origin/${ref}"
  fi

  # Final best-effort fetch for raw refs (e.g. refs/pull/* or specific SHAs).
  git -C "${repo_root}" fetch --no-tags origin "${ref}" >/dev/null 2>&1 || true

  if git -C "${repo_root}" rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
    git -C "${repo_root}" rev-parse --verify "${ref}^{commit}"
    return 0
  fi

  if [[ -n "${alternate_ref}" ]] && git -C "${repo_root}" rev-parse --verify "${alternate_ref}^{commit}" >/dev/null 2>&1; then
    git -C "${repo_root}" rev-parse --verify "${alternate_ref}^{commit}"
    return 0
  fi

  return 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 2 ]]; then
  usage
  exit 2
fi

base_ref="$1"
test_ref="$2"
shift 2
compare_paths_args=("$@")

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${ci_dir}/../.." && pwd)"

if [[ ! -f "${repo_root}/cccl-version.json" ]]; then
  die "This script must run from a CCCL checkout."
fi

if ! base_commit="$(resolve_ref_to_commit "${repo_root}" "${base_ref}")"; then
  die "Unable to resolve base ref: ${base_ref}"
fi
if ! test_commit="$(resolve_ref_to_commit "${repo_root}" "${test_ref}")"; then
  die "Unable to resolve test ref: ${test_ref}"
fi

base_short_sha="$(git -C "${repo_root}" rev-parse --short=12 "${base_commit}")"
test_short_sha="$(git -C "${repo_root}" rev-parse --short=12 "${test_commit}")"
base_label="$(display_label_for_ref "${base_ref}" "${base_short_sha}")"
test_label="$(display_label_for_ref "${test_ref}" "${test_short_sha}")"

worktree_root="$(mktemp -d "/tmp/cccl-bench-worktrees-XXXXXX")"
base_path="${worktree_root}/base"
test_path="${worktree_root}/test"

cleanup() {
  git -C "${repo_root}" worktree remove --force "${base_path}" >/dev/null 2>&1 || true
  git -C "${repo_root}" worktree remove --force "${test_path}" >/dev/null 2>&1 || true
  rm -rf "${worktree_root}"
}
trap cleanup EXIT

echo "Creating worktree for base ref: ${base_ref}"
git -C "${repo_root}" worktree add --detach "${base_path}" "${base_commit}" >/dev/null

echo "Creating worktree for test ref: ${test_ref}"
git -C "${repo_root}" worktree add --detach "${test_path}" "${test_commit}" >/dev/null

compare_cmd=(
  "${ci_dir}/compare_paths.sh"
  "${base_path}"
  "${test_path}"
  "${compare_paths_args[@]}"
)

CCCL_BENCH_BASE_LABEL="${base_label}" \
CCCL_BENCH_TEST_LABEL="${test_label}" \
  "${compare_cmd[@]}"
