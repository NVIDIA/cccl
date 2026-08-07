#!/usr/bin/env bash

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

build_target=""
codegen_tests=false
common_args=()
while [[ $# -ne 0 ]]; do
  case "$1" in
    -codegen-tests)
      codegen_tests=true
      shift
      ;;
    -target)
      if [[ $# -lt 2 ]]; then
        echo "Error: -target requires a value" >&2
        exit 1
      fi
      build_target="$2"
      shift 2
      ;;
    *)
      common_args+=("$1")
      shift
      ;;
  esac
done
set -- "${common_args[@]}"

# shellcheck source=ci/build_common.sh
source "${ci_dir}/build_common.sh"

print_environment_details

PRESET="libcudacxx"
if $codegen_tests; then
  PRESET="libcudacxx-codegen-filecheck"
fi

CMAKE_OPTIONS=("-DCMAKE_CXX_STANDARD=${CXX_STANDARD}" "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}")

if [[ -n "$build_target" ]]; then
  configure_preset "$PRESET" "$PRESET" "${CMAKE_OPTIONS[@]}"
  if ! $CONFIGURE_ONLY; then
    build_preset "$PRESET" "$PRESET" --target "$build_target"
  fi
  print_time_summary
  exit 0
fi

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
