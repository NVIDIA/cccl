#!/bin/bash

set -euo pipefail

CPU_ONLY=
GPU_ONLY=

new_args=$(ci/util/extract_switches.sh -cpu-only -gpu-only -- "$@")
eval set -- ${new_args}
while true; do
  case "$1" in
  -cpu-only)
    CPU_ONLY=true
    shift
    ;;
  -gpu-only)
    GPU_ONLY=true
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

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_thrust.sh "$@"

# Default: run all
PRESETS=(
  "thrust-cpu-cpp$CXX_STANDARD"
  "thrust-gpu-cpp$CXX_STANDARD"
)
GPU_REQUIRED="true"

if [ -n "$CPU_ONLY" ]; then
  PRESETS=("thrust-cpu-cpp$CXX_STANDARD")
  GPU_REQUIRED="false"
elif [ -n "$GPU_ONLY" ]; then
  PRESETS=("thrust-gpu-cpp$CXX_STANDARD")
  GPU_REQUIRED="true"
fi

for PRESET in ${PRESETS[@]}; do
  test_preset "Thrust (${PRESET})" ${PRESET} ${GPU_REQUIRED}
done

print_time_summary
