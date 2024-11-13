#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="cudax-cpp$CXX_STANDARD"

CMAKE_OPTIONS=""

configure_and_build_preset "CUDA Experimental" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
