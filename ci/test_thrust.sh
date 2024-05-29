#!/bin/bash

set -euo pipefail

CPU_ONLY=false
GPU_ONLY=false

ci_dir=$(dirname "$0")

new_args=$("${ci_dir}/util/extract_switches.sh" -cpu-only -gpu-only -- "$@")
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

source "${ci_dir}/build_common.sh"

print_environment_details

./build_thrust.sh "$@"

if $CPU_ONLY; then
  PRESETS=("thrust-cpu-cpp$CXX_STANDARD")
  GPU_REQUIRED=false
elif $GPU_ONLY; then
  PRESETS=("thrust-gpu-cpp$CXX_STANDARD")
  GPU_REQUIRED=true
else
  PRESETS=("thrust-cpp$CXX_STANDARD")
  GPU_REQUIRED=true
fi

for PRESET in ${PRESETS[@]}; do
  test_preset "Thrust (${PRESET})" ${PRESET} ${GPU_REQUIRED}
done

print_time_summary
