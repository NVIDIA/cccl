#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Needs `gh` (or docker, for a local build), which the minimal container lacks.
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Run the payload in a minimal sibling container, except in `sysctk` mode or
# when CCCL_MINIMAL_CONTAINER=0. See "Testing Python in a minimal container" in
# docs/infrastructure/ci/references/ci_overview.rst.
readonly payload="ci/util/python/run_compute_tests.sh"
if [[ "${ctk_mode,,}" != "sysctk" && "${CCCL_MINIMAL_CONTAINER:-1}" != "0" ]]; then
  exec "$ci_dir/util/python/run_in_minimal_container.sh" "${payload}" "$@"
else
  exec "${repo_root}/${payload}" "$@"
fi
