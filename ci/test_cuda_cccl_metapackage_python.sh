#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
pin_cuda_toolkit "${ctk_mode}"
setup_python_env "${py_version}"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Constrain cuda-compute to the coordinated local wheel without making it a
# direct install requirement. This proves the metapackage resolves compute
# transitively while preventing an equal-version index candidate from winning;
# index access remains available for third-party dependencies.
wheelhouse_dir="${repo_root}/wheelhouse"
mapfile -t cuda_cccl_wheels < <(
  find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'cuda_cccl-*.whl' -print | sort
)
mapfile -t cuda_compute_wheels < <(
  find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'cuda_compute-*.whl' -print | sort
)
if [[ "${#cuda_cccl_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one cuda-cccl wheel in ${wheelhouse_dir}; found ${#cuda_cccl_wheels[@]}." >&2
  exit 1
fi
if [[ "${#cuda_compute_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one cuda-compute wheel in ${wheelhouse_dir}; found ${#cuda_compute_wheels[@]}." >&2
  exit 1
fi
CUDA_CCCL_WHEEL_PATH="${cuda_cccl_wheels[0]}"
CUDA_COMPUTE_WHEEL_PATH="${cuda_compute_wheels[0]}"
cuda_compute_wheel_uri=$(python -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().as_uri())' \
  "${CUDA_COMPUTE_WHEEL_PATH}")
constraints_file=$(mktemp)
trap 'rm -f "${constraints_file}"' EXIT
printf 'cuda-compute @ %s\n' "${cuda_compute_wheel_uri}" > "${constraints_file}"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"
python -m pip install --constraint "${constraints_file}" \
  --find-links "${wheelhouse_dir}" \
  "${CUDA_CCCL_WHEEL_PATH}[minimal-${ctk_flavor}${cuda_major_version}]"
python -m pip check
python - "${CUDA_COMPUTE_WHEEL_PATH}" <<'PY'
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import cuda.compute

compute_distribution = importlib.metadata.distribution("cuda-compute")
direct_url_text = compute_distribution.read_text("direct_url.json")
if direct_url_text is None:
    raise RuntimeError("cuda-compute is missing direct_url.json provenance")
compute_url = json.loads(direct_url_text)["url"]
expected_compute_url = Path(sys.argv[1]).resolve().as_uri()
if compute_url != expected_compute_url:
    raise RuntimeError(
        f"cuda-compute came from {compute_url}, expected {expected_compute_url}"
    )

metapackage_version = importlib.metadata.version("cuda-cccl")
compute_version = importlib.metadata.version("cuda-compute")
if metapackage_version != compute_version:
    raise RuntimeError(
        f"cuda-cccl {metapackage_version} != cuda-compute {compute_version}"
    )
if importlib.util.find_spec("cuda.cccl") is not None:
    raise RuntimeError("cuda-cccl must not install a cuda.cccl module")
PY
