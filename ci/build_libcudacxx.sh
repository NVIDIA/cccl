#!/usr/bin/env bash

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=ci/build_common.sh
source "${ci_dir}/build_common.sh"

print_environment_details

PRESET="libcudacxx"
CMAKE_OPTIONS=("-DCMAKE_CXX_STANDARD=${CXX_STANDARD}" "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}")

upload_test_artifacts=false
if [[ -n "${GITHUB_ACTIONS:-}" ]] && "${ci_dir}/util/workflow/has_consumers.sh"; then
  upload_test_artifacts=true
  export LIT_OPTS="${LIT_OPTS:+${LIT_OPTS} }-Dtest_executable_mode=build"
fi

configure_and_build_preset libcudacxx "$PRESET" "${CMAKE_OPTIONS[@]}"

if $upload_test_artifacts; then
  run_command "📦  Packaging test artifacts" "${ci_dir}/upload_libcudacxx_test_artifacts.sh"
fi

print_time_summary
