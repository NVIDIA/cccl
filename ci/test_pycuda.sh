#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

readonly prefix="${BUILD_DIR}/python/"
export PYTHONPATH="${prefix}:${PYTHONPATH:-}"

pushd ../python/cuda >/dev/null

run_command "⚙️  Pip install cuda" pip install --force-reinstall --target "${prefix}" .[test]
run_command "🚀  Pytest cuda" python -m pytest -v ./tests

popd >/dev/null

print_time_summary
