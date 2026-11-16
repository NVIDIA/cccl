#!/usr/bin/env bash

# Check that HEAD has no ABI collision with the latest stable CCCL release.
#
# CCCL is header-only, so a library built against CCCL carries its own copies of CCCL's inline
# functions, function-local statics and vtables. When two such libraries built against
# different CCCL releases are linked into one binary, the linker keeps one definition per
# mangled name. A symbol that keeps its mangled name but changes its definition between
# releases therefore breaks that binary. See ci/abi/README.md.
#
# Exit status is 0 when no difference is found, 1 otherwise.
set -euo pipefail

usage()
{
  cat <<EOF
Usage: $0 [options]

Options:
  -base-ref <ref>     Release to compare against. Default: the newest vX.Y.Z tag.
  -test-ref <ref>     Revision or directory to check. Default: HEAD.
  -output-dir <path>  Directory for report.json. Default: "<cwd>/abi-artifacts".
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base_ref=""
test_ref="HEAD"
output_dir="${PWD}/abi-artifacts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -base-ref) base_ref="$2"; shift 2 ;;
    -test-ref) test_ref="$2"; shift 2 ;;
    -output-dir) output_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ -z "${base_ref}" ]]; then
  base_ref="$(git -C "${repo_root}" tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)"
  if [[ -z "${base_ref}" ]]; then
    echo "No vX.Y.Z tag found; fetch tags or pass -base-ref." >&2
    exit 2
  fi
fi

echo "Comparing ${test_ref} against ${base_ref}"
mkdir -p "${output_dir}"

# cuda::experimental has no ABI guarantee. CUB and Thrust put every symbol in an inline
# namespace that encodes their version and target architectures, so they cannot collide
# across releases. The match is limited to the text before the first '(' so that a `cuda::`
# function that merely takes a Thrust or CUB argument is still checked.
exclude='^(?:[^(]*\s)?(?:cuda::experimental|thrust|cub)::'

python3 "${repo_root}/ci/abi/cccl_abi_diff.py" \
  "${base_ref}" "${test_ref}" \
  --repo "${repo_root}" \
  --cache-dir "${output_dir}/cache" \
  --exclude "${exclude}" \
  --json "${output_dir}/report.json"
