#!/bin/bash

source "$(dirname "$0")/build_common.sh"

./build_thrust.sh "$@"

ctest --test-dir ${BUILD_DIR} --output-on-failure 

echo "Thrust test complete"

