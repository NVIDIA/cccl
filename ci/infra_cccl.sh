#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="cccl-infra"

CMAKE_OPTIONS=""

GPU_REQUIRED="false"

if [ -n "${GITHUB_SHA:-}" ]; then
  CMAKE_OPTIONS="$CMAKE_OPTIONS -DCCCL_EXAMPLE_CPM_TAG=${GITHUB_SHA}"
fi

configure_preset "CCCL Infra" "$PRESET" "$CMAKE_OPTIONS"
test_preset "CCCL Infra" "$PRESET" "$GPU_REQUIRED"

print_time_summary
