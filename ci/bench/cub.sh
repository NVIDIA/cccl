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

Wrapper for ci/bench/compare_git_refs.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 2 ]]; then
  usage
  die "Expected at least <base-ref> and <test-ref>."
fi

bench_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${bench_dir}/compare_git_refs.sh" "$@"
