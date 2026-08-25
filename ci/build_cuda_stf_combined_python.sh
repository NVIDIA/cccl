#!/usr/bin/env bash
set -euo pipefail

# Combined producer for the cuda.stf._experimental test lane.
#
# cuda-stf tests must run against the *current branch's* cuda-cccl as well as
# the current branch's cuda-stf. The workflow model lets a consumer job depend
# on only a single producer, so this one producer builds and uploads BOTH
# wheels under distinct artifact names (via get_wheel_artifact_name.sh):
#
#   * cuda-cccl  -> wheel-cccl-stf-<os>-<arch>-py<ver>  (CCCL_WHEEL_KIND=cccl-stf)
#   * cuda-stf   -> wheel-stf-<os>-<arch>-py<ver>       (CCCL_WHEEL_KIND=stf)
#
# The cuda-cccl wheel uses the dedicated 'cccl-stf' kind so it does NOT collide
# with the 'wheel-cccl-...' artifact produced by the regular build_py_wheel job
# (which also runs in project 'python' with the same py/os/arch).
#
# ci/test_cuda_stf_python.sh then downloads and installs both local wheels in a
# single resolver invocation. Build cuda-cccl first: the cuda-stf build leaves
# any co-located cuda_cccl wheel in wheelhouse/ untouched.

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1

echo "::group::⚒️ Building current-branch cuda-cccl wheel"
CCCL_WHEEL_KIND=cccl-stf "$ci_dir/build_cuda_cccl_python.sh" "$@"
echo "::endgroup::"

echo "::group::⚒️ Building current-branch cuda-stf wheel"
"$ci_dir/build_cuda_stf_python.sh" "$@"
echo "::endgroup::"
