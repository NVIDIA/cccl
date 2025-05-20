#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

if [[ -e /usr/local/cuda/targets/x86_64-linux/include ]]; then
    ls -l /usr/local/cuda/targets/x86_64-linux/include
fi

PRESET="cccl-c-parallel"

CMAKE_OPTIONS=""

configure_and_build_preset "CCCL C Parallel Library" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
