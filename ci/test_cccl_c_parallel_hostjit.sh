#!/bin/bash

# shellcheck source=ci/build_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

./build_cccl_c_parallel_hostjit.sh "$@"

PRESET="cccl-c-parallel-hostjit"

test_preset "CCCL C Parallel Library (HostJIT)" "$PRESET"

print_time_summary
