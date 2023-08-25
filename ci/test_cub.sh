#!/bin/bash

source "$(dirname "$0")/build_common.sh"

./build_cub.sh "$@"

PRESET="cub-cpp$CXX_STANDARD"

test_preset CUB "${PRESET}"
