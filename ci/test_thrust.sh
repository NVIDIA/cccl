#!/bin/bash

source "$(dirname "$0")/build_common.sh"

./build_thrust.sh "$@"

PRESET="thrust-cpp$CXX_STANDARD"

pushd ..
ctest --preset=$PRESET
popd

echo "Thrust test complete"
