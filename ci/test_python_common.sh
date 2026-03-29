set -euo pipefail

function list_environment {
  begin_group "⚙️ Existing site-packages"
  pip freeze
  end_group "⚙️ Existing site-packages"
}

function run_tests {
  module=$1

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
  run_command "🚀  Pytest ${module}" pytest -n ${PARALLEL_LEVEL} -v ./tests
  deactivate

  popd >/dev/null

  print_time_summary
}
