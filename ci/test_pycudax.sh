#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

readonly prefix="${BUILD_DIR}/python/"
export PYTHONPATH="${prefix}:${PYTHONPATH:-}"

pushd ../python/cudax >/dev/null

run_command "⚙️  Pip install cudax" pip install --force-reinstall --target "${prefix}" .
run_command "🚀  Pytest cudax" python -m pytest -v ./tests

popd >/dev/null

print_time_summary
