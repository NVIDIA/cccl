#!/bin/bash

source "$(dirname "$0")/build_common.sh"

./build_cub.sh "$@"

PRESET="ci-cub-cpp$CXX_STANDARD"

pushd ..
ctest --preset=$PRESET
popd

echo "CUB test complete"
