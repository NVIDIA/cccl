#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

PRESET="nvbench-helper"

CMAKE_OPTIONS=""

GPU_REQUIRED="true"

configure_and_build_preset "NVBench Helper" "$PRESET" "$CMAKE_OPTIONS"
test_preset "NVBench Helper" "$PRESET" "$GPU_REQUIRED"

print_time_summary
