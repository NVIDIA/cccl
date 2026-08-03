#!/usr/bin/env bash

set -euo pipefail
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
# Pin cuda-toolkit to the container's CTK minor and set cuda_version /
# cuda_major_version (-ctk-mode latest opts out). See pyenv_helper.sh.
pin_cuda_toolkit "${ctk_mode}"

# Setup Python environment
setup_python_env "${py_version}"

# Fetch or build the cuda_cccl wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install only the header distribution to verify that it does not require the
# compute wheel, metapackage, or CUDA Python bindings.
wheelhouse_dir="${repo_root}/wheelhouse"
CCCL_HEADERS_WHEEL_PATH="$(ls "${wheelhouse_dir}"/cccl_headers-*.whl)"
python -m pip install --find-links "${wheelhouse_dir}" "${CCCL_HEADERS_WHEEL_PATH}" pytest pytest-xdist
python -m pip check
python - <<'PY'
import importlib.metadata

for distribution in ("cuda-compute", "cuda-cccl"):
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError(f"{distribution} must not be installed by cccl-headers")
PY

cd "${repo_root}/python/cccl_headers"
python -m pytest -n auto -v tests/
