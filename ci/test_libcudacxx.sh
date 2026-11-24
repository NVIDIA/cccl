#!/usr/bin/env bash

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=ci/build_common.sh
source "${ci_dir}/build_common.sh"

print_environment_details

PRESET="libcudacxx"
CMAKE_OPTIONS=("-DCMAKE_CXX_STANDARD=${CXX_STANDARD}" "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}")

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  producer_id="$(util/workflow/get_producer_id.sh)"
  artifact="z_libcudacxx-test-artifacts-${DEVCONTAINER_NAME:?}-$producer_id"
  run_command "📦  Unpacking artifact '$artifact'" \
    "${ci_dir}/util/artifacts/download_packed.sh" "$artifact" /home/coder/cccl/
else
  "${ci_dir}/build_libcudacxx.sh" "$@"
  configure_preset libcudacxx "$PRESET" "${CMAKE_OPTIONS[@]}"
fi

test_preset "libcudacxx (CTest)" "libcudacxx-ctest"

sccache -z > /dev/null || :

lit_test_name="libcudacxx (lit)"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  export LIT_OPTS="${LIT_OPTS:+${LIT_OPTS} }-Dtest_executable_mode=replay"
  lit_test_name="libcudacxx (lit replay)"
fi
test_preset "${lit_test_name}" "libcudacxx-lit"

sccache --show-adv-stats || :

print_time_summary
