#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_thrust.sh "$@"

PRESET="thrust-cpp$CXX_STANDARD"

test_preset "Thrust" ${PRESET}

print_time_summary
