#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

readonly prefix="${BUILD_DIR}/python/"
export PYTHONPATH="${prefix}:${PYTHONPATH:-}"

pushd ../python/cuda_cooperative >/dev/null

run_command "⚙️  Pip install cuda_cooperative" pip install --force-reinstall --upgrade --target "${prefix}" .[test]
run_command "🚀  Pytest cuda_cooperative" python -m pytest -v ./tests

popd >/dev/null

pushd ../python/cuda_parallel >/dev/null

run_command "⚙️  Pip install cuda_parallel" pip install --force-reinstall --upgrade --target "${prefix}" .[test]
run_command "🚀  Pytest cuda_parallel" python -m pytest -v ./tests

popd >/dev/null

print_time_summary
