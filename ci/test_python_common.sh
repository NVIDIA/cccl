set -euo pipefail

python_common_ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_common_repo_root="$(cd "${python_common_ci_dir}/.." && pwd)"

function list_environment {
  begin_group "⚙️ Existing site-packages"
  pip freeze
  end_group "⚙️ Existing site-packages"
}

function run_tests {
  module=$1

  pushd "${python_common_repo_root}/python/${module}" >/dev/null

  TEMP_VENV_DIR="/tmp/${module}_venv"
  rm -rf "${TEMP_VENV_DIR}"
  python -m venv "${TEMP_VENV_DIR}"
  # shellcheck disable=SC1091
  . "${TEMP_VENV_DIR}/bin/activate"
  cat > /tmp/cuda-cccl_constraints.txt <<EOF
cccl-headers @ file://${python_common_repo_root}/python/cccl_headers
cuda-compute @ file://${python_common_repo_root}/python/cuda_compute
cuda-cccl @ file://${python_common_repo_root}/python/cuda_cccl
EOF
  run_command "⚙️  Pip install ${module}" pip install -c /tmp/cuda-cccl_constraints.txt ".[test]"
  begin_group "⚙️ ${module} site-packages"
  pip freeze
  end_group "⚙️ ${module} site-packages"
  run_command "🚀  Pytest ${module}" pytest -n "${PARALLEL_LEVEL:-$(nproc --all --ignore=1)}" -v ./tests
  deactivate

  popd >/dev/null

  print_time_summary
}
