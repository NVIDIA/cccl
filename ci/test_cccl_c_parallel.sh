#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_cccl_c_parallel.sh "$@"

PRESET="cccl-c-parallel"

test_preset "CCCL C Parallel Library" ${PRESET}

print_time_summary
