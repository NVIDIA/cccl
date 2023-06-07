#!/bin/bash

set -xeuo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

./build_cub.sh "$@"

ctest --test-dir ../build --output-on-failure

echo "CUB test complete"