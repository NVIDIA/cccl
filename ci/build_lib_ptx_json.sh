#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="cccl-lib-ptx-json"

CMAKE_OPTIONS=""

configure_and_build_preset "CCCL PTX-JSON Library" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
