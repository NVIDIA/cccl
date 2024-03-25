#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_cub.sh "$@"

PRESET="cub-cpp$CXX_STANDARD"

test_preset CUB "${PRESET}"

print_time_summary
