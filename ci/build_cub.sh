#!/bin/bash

set -euo pipefail

NO_LID=false
LID0=false
LID1=false
LID2=false
ARTIFACT_TAGS=()

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

new_args=$("${ci_dir}/util/extract_switches.sh" \
  -no-lid \
  -lid0 \
  -lid1 \
  -lid2 \
  -- "$@")

eval set -- ${new_args}
while true; do
  case "$1" in
  -no-lid)
    ARTIFACT_TAGS+=("no_lid")
    NO_LID=true
    shift
    ;;
  -lid0)
    ARTIFACT_TAGS+=("lid_0")
    LID0=true
    shift
    ;;
  -lid1)
    ARTIFACT_TAGS+=("lid_1")
    LID1=true
    shift
    ;;
  -lid2)
    ARTIFACT_TAGS+=("lid_2")
    LID2=true
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

source "${ci_dir}/build_common.sh"

print_environment_details

ENABLE_CCCL_BENCHMARKS="false"
ENABLE_CUB_RDC="false"

if [[ "$CUDA_COMPILER" == *nvcc* ]]; then
    ENABLE_CUB_RDC="true"
    NVCC_VERSION=$($CUDA_COMPILER --version | grep release | awk '{print $6}' | cut -c2-)
    if [[ -n "${DISABLE_CUB_BENCHMARKS}" ]]; then
        echo "Benchmarks have been forcefully disabled."
    else
        ENABLE_CCCL_BENCHMARKS="true"
        echo "nvcc version is $NVCC_VERSION. Building CUB benchmarks."
    fi
else
    echo "Not building with NVCC, disabling RDC and benchmarks."
fi

if [[ "$HOST_COMPILER" == *icpc* || "$HOST_COMPILER" == *nvhpc* ]]; then
    ENABLE_CCCL_BENCHMARKS="false"
fi

PRESET="cub"
if $NO_LID; then
    PRESET="cub-nolid"
elif $LID0; then
    PRESET="cub-lid0"
elif $LID1; then
    PRESET="cub-lid1"
elif $LID2; then
    PRESET="cub-lid2"
fi

CMAKE_OPTIONS=(
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD
    -DCMAKE_CUDA_STANDARD=$CXX_STANDARD
    -DCCCL_ENABLE_BENCHMARKS=$ENABLE_CCCL_BENCHMARKS
    -DCUB_ENABLE_RDC_TESTS=$ENABLE_CUB_RDC
)

configure_and_build_preset "CUB" "$PRESET" "${CMAKE_OPTIONS[*]}"

# Create test artifacts:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    if [[ ${#ARTIFACT_TAGS[@]} -gt 0 ]]; then
        run_command "📦  Packaging test artifacts" \
            "${ci_dir}/upload_cub_test_artifacts.sh" \
            "${ARTIFACT_TAGS[@]}"
    else
        run_command "📦  Packaging test artifacts" "${ci_dir}/upload_cub_test_artifacts.sh"
    fi
fi

print_time_summary
