#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_cudax.sh "$@"

PRESET="cudax-cpp$CXX_STANDARD"

test_preset "CUDA Experimental" ${PRESET}

print_time_summary
