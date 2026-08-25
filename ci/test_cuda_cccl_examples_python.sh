#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Derived, not hardcoded to the devcontainer workspace: for `devcontainer: false`
# lanes this half runs on the CI runner, where the checkout lives elsewhere.
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

# Run the test payload. Lanes with `devcontainer: false` in ci/matrix.yaml run it
# in a container holding nothing but Python, so that any reliance on a host
# compiler or a system CUDA toolkit fails loudly. Every other lane runs it right
# here, in the devcontainer. JOB_DEVCONTAINER is set by the CI action.
readonly payload="ci/util/python/run_examples_tests.sh"
if [[ "${JOB_DEVCONTAINER:-true}" == "false" ]]; then
  exec "$ci_dir/util/python/run_in_minimal_container.sh" "${payload}" "$@"
else
  exec "${repo_root}/${payload}" "$@"
fi
