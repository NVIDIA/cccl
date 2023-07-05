#!/bin/bash

source "$(dirname "$0")/build_common.sh"

./build_cub.sh "$@"

ctest --test-dir ${BUILD_DIR} --output-on-failure 

echo "CUB test complete"