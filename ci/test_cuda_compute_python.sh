#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Fetch or build the cuda_cccl wheel. This needs `gh` (or docker, for a local
# build) -- tooling the minimal test container deliberately does not have -- so
# it happens out here, before the test payload runs.
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Run the test payload in a minimal sibling container (see
# run_in_minimal_container.sh for what that buys). `sysctk` is the exception:
# that mode exists to test against a system-provided CUDA toolkit, which only
# the devcontainer has. Set CCCL_MINIMAL_CONTAINER=0 to stay here as well --
# useful locally, or to compare against the devcontainer environment.
readonly payload="ci/util/python/run_compute_tests.sh"
if [[ "${ctk_mode,,}" != "sysctk" && "${CCCL_MINIMAL_CONTAINER:-1}" != "0" ]]; then
  exec "$ci_dir/util/python/run_in_minimal_container.sh" "${payload}" "$@"
else
  exec "${repo_root}/${payload}" "$@"
fi
