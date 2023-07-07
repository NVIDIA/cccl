#!/bin/bash

set -xeuo pipefail

if [ -n ${LIBCUDACXX_USE_NVRTC+x} ]; then
    echo "The LIBCUDACXX_USE_NVRTC configuration is only used for libcu++. Not running any tests."
    exit 0
fi

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

./build_thrust.sh "$@"

ctest --test-dir ../build --output-on-failure --timeout 15

echo "Thrust test complete"

