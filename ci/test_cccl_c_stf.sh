#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

./build_cccl_c_stf.sh "$@"

PRESET="cccl-c-stf"

test_preset "CCCL C Parallel Library" ${PRESET}

print_time_summary
