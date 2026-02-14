#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

./build_cudax.sh "$@"

PRESET="cudax"

test_preset "CUDA Experimental" ${PRESET}

print_time_summary
