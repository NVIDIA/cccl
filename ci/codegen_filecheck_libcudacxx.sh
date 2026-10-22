#!/usr/bin/env bash

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

build_target=""
common_args=()
while [[ $# -ne 0 ]]; do
  case "$1" in
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

if [[ -z "$build_target" ]]; then
  echo "Error: -target is required" >&2
  exit 1
fi

# shellcheck source=ci/build_common.sh
source "${ci_dir}/build_common.sh"

print_environment_details

PRESET="libcudacxx-codegen-filecheck"
CMAKE_OPTIONS=("-DCMAKE_CXX_STANDARD=${CXX_STANDARD}" "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}")

configure_preset "$PRESET" "$PRESET" "${CMAKE_OPTIONS[@]}"
if ! $CONFIGURE_ONLY; then
  # Report every FileCheck failure in the requested suite.
  build_preset "$PRESET" "$PRESET" --target "$build_target" -- -k 0
fi

print_time_summary
