#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

begin_group "⚙️ Existing site-packages"
pip freeze
end_group "⚙️ Existing site-packages"

for module in cuda_parallel cuda_cooperative; do

  pushd "../python/${module}" >/dev/null

  TEMP_VENV_DIR="/tmp/${module}_venv"
  rm -rf "${TEMP_VENV_DIR}"
  python -m venv "${TEMP_VENV_DIR}"
  . "${TEMP_VENV_DIR}/bin/activate"
  echo 'cuda-cccl @ file:///home/coder/cccl/python/cuda_cccl' > /tmp/cuda-cccl_constraints.txt
  run_command "⚙️  Pip install ${module}" pip install -c /tmp/cuda-cccl_constraints.txt .[test]
  begin_group "⚙️ ${module} site-packages"
  pip freeze
  end_group "⚙️ ${module} site-packages"
  run_command "🚀  Pytest ${module}" python -m pytest -v ./tests
  deactivate

  popd >/dev/null

done

print_time_summary
