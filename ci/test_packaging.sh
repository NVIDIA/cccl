#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

PRESET="packaging"

CMAKE_OPTIONS=""

GPU_REQUIRED="true"

if [ -n "${GITHUB_SHA:-}" ]; then
  CMAKE_OPTIONS="$CMAKE_OPTIONS -DCCCL_EXAMPLE_CPM_TAG=${GITHUB_SHA}"
fi

configure_and_build_preset "Packaging" "$PRESET" "$CMAKE_OPTIONS"
test_preset "Packaging" "$PRESET" "$GPU_REQUIRED"

print_time_summary
