#!/bin/bash

set -euo pipefail

CPU_ONLY=false
GPU_ONLY=false

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

new_args=$("${ci_dir}/util/extract_switches.sh" -cpu-only -gpu-only -- "$@")
eval set -- ${new_args}
while true; do
  case "$1" in
  -cpu-only)
    ARTIFACT_TAG="test_cpu"
    CPU_ONLY=true
    shift
    ;;
  -gpu-only)
    ARTIFACT_TAG="test_gpu"
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

if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  ./build_thrust.sh "$@"
else
  run_command "📦  Unpacking test artifacts" \
    "${ci_dir}/util/artifacts/download_packed.sh" \
      "z_thrust-test-artifacts-$DEVCONTAINER_NAME-$(util/workflow/get_producer_id.sh)-$ARTIFACT_TAG" \
      /home/coder/cccl/
fi

if $CPU_ONLY; then
  PRESETS=("thrust-cpu")
  GPU_REQUIRED=false
elif $GPU_ONLY; then
  PRESETS=("thrust-gpu")
  GPU_REQUIRED=true
else
  PRESETS=("thrust")
  GPU_REQUIRED=true
fi

for PRESET in ${PRESETS[@]}; do
  test_preset "Thrust (${PRESET})" ${PRESET} ${GPU_REQUIRED}
done

print_time_summary
