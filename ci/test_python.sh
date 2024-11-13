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

# Temporarily install the package twice to populate include directory as part of the first installation
# and to let manifest discover these includes during the second installation. Do not forget to remove the
# second installation after https://github.com/NVIDIA/cccl/issues/2281 is addressed.
run_command "⚙️  Pip install cuda_parallel once" pip install --force-reinstall --upgrade --target "${prefix}" .[test]
run_command "⚙️  Pip install cuda_parallel twice" pip install --force-reinstall --upgrade --target "${prefix}" .[test]
run_command "🚀  Pytest cuda_parallel" python -m pytest -v ./tests

popd >/dev/null

print_time_summary
