#!/bin/bash

set -euo pipefail

ci_dir="$(dirname "${BASH_SOURCE[0]}")"
cccl_dir="$(realpath "${ci_dir}/..")"
source "${ci_dir}/build_common.sh"
cd "${ci_dir}"

print_environment_details

PRESET="packaging"

CMAKE_OPTIONS=""

GPU_REQUIRED="true"

CMAKE_OPTIONS=("-DCCCL_EXAMPLE_CPM_REPOSITORY=${cccl_dir}")

# Local -- build against the current repo's HEAD commit:
if [ -z "${GITHUB_ACTIONS:-}" ]; then
  CMAKE_OPTIONS+=("-DCCCL_EXAMPLE_CPM_TAG=HEAD")
else
  CMAKE_OPTIONS+=("-DCCCL_EXAMPLE_CPM_TAG=${GITHUB_SHA}")
fi

configure_and_build_preset "Packaging" "$PRESET" "${CMAKE_OPTIONS[@]}"
test_preset "Packaging" "$PRESET" "$GPU_REQUIRED"

print_time_summary
