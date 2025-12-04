#!/bin/bash

set -euo pipefail

ci_dir="$(dirname "${BASH_SOURCE[0]}")"
cccl_dir="$(realpath "${ci_dir}/..")"

MIN_CMAKE=false
minimum_cmake_version=3.18.0

new_args=$("${ci_dir}/util/extract_switches.sh" -min-cmake -- "$@")
eval set -- ${new_args}
while true; do
  case "$1" in
  -min-cmake)
    MIN_CMAKE=true
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "Unknown argument: $1"
    exit 1
    ;;
  esac
done

if $MIN_CMAKE; then
  echo "Installing minimum CMake version v${minimum_cmake_version}..."
  wget -q \
    https://github.com/Kitware/CMake/releases/download/v${minimum_cmake_version}/cmake-${minimum_cmake_version}-Linux-x86_64.sh \
    -O /tmp/cmake-install.sh
  prefix=/tmp/cmake-${minimum_cmake_version}
  mkdir -p ${prefix}
  bash /tmp/cmake-install.sh --skip-license --prefix=${prefix}
  export MIN_CTEST_COMMAND="${prefix}/bin/ctest"
fi

# Needs to happen after cmake is installed:
source "${ci_dir}/build_common.sh"
cd "${ci_dir}"

print_environment_details

PRESET="packaging"

CMAKE_OPTIONS=""

GPU_REQUIRED="true"

CMAKE_OPTIONS=("-DCCCL_EXAMPLE_CPM_REPOSITORY=${cccl_dir}")

# Local -- build against the current repo's HEAD commit:
if [ -z "${GITHUB_ACTIONS:-}" ]; then
  CMAKE_OPTIONS+=("-DCCCL_EXAMPLE_CPM_TAG=HEAD")
else
  CMAKE_OPTIONS+=("-DCCCL_EXAMPLE_CPM_TAG=${GITHUB_SHA}")
fi

if [[ -n "${MIN_CTEST_COMMAND:-}" ]]; then
  CMAKE_OPTIONS+=("-DCCCL_EXAMPLE_CTEST_COMMAND=${MIN_CTEST_COMMAND}")
fi

configure_and_build_preset "Packaging" "$PRESET" "${CMAKE_OPTIONS[@]}"
test_preset "Packaging" "$PRESET" "$GPU_REQUIRED"

print_time_summary
