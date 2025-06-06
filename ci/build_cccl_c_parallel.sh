#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

PRESET="cccl-c-parallel"

CMAKE_OPTIONS=""

configure_and_build_preset "CCCL C Parallel Library" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
