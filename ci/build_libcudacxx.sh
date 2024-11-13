#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_and_build_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
