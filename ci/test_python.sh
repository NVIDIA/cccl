#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

begin_group "⚙️ Existing site-packages"
pip freeze
end_group "⚙️ Existing site-packages"

pushd ../python/cuda_cooperative >/dev/null

rm -rf /tmp/cuda_cooperative_venv
python -m venv /tmp/cuda_cooperative_venv
. /tmp/cuda_cooperative_venv/bin/activate
echo 'cuda-cccl @ file:///home/coder/cccl/python/cuda_cccl' > /tmp/cuda-cccl_constraints.txt
run_command "⚙️  Pip install cuda_cooperative" pip install -c /tmp/cuda-cccl_constraints.txt .[test]
begin_group "⚙️ cuda-cooperative site-packages"
pip freeze
end_group "⚙️ cuda-cooperative site-packages"
run_command "🚀  Pytest cuda_cooperative" python -m pytest -v ./tests
deactivate

popd >/dev/null

pushd ../python/cuda_parallel >/dev/null

rm -rf /tmp/cuda_parallel_venv
python -m venv /tmp/cuda_parallel_venv
. /tmp/cuda_parallel_venv/bin/activate
echo 'cuda-cccl @ file:///home/coder/cccl/python/cuda_cccl' > /tmp/cuda-cccl_constraints.txt
run_command "⚙️  Pip install cuda_parallel" pip install -c /tmp/cuda-cccl_constraints.txt .[test]
begin_group "⚙️ cuda-parallel site-packages"
pip freeze
end_group "⚙️ cuda-parallel site-packages"
run_command "🚀  Pytest cuda_parallel" python -m pytest -v ./tests
deactivate

popd >/dev/null

print_time_summary
