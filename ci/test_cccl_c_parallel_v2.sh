#!/usr/bin/env bash

# shellcheck source=ci/build_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

./build_cccl_c_parallel_v2.sh "$@"

PRESET="cccl-c-parallel-v2"

test_preset "CCCL C Parallel Library v2 (HostJIT)" "$PRESET"

print_time_summary
