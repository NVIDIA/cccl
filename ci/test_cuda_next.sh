#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_cuda_next.sh "$@"

PRESET="cuda-next-cpp$CXX_STANDARD"

test_preset "CudaNext" ${PRESET}

print_time_summary
