#!/bin/bash

set -euo pipefail

NO_LID=false
LID0=false
LID1=false
LID2=false
LIMITED=false
COMPUTE_SANITIZER=false

ci_dir=$(dirname "$0")

new_args=$("${ci_dir}/util/extract_switches.sh" \
  -no-lid \
  -lid0 \
  -lid1 \
  -lid2 \
  -limited \
  -compute-sanitizer-memcheck \
  -compute-sanitizer-racecheck \
  -compute-sanitizer-initcheck \
  -compute-sanitizer-synccheck \
  -- "$@")

eval set -- ${new_args}
while true; do
  case "$1" in
  -no-lid)
    NO_LID=true
    shift
    ;;
  -lid0)
    LID0=true
    shift
    ;;
  -lid1)
    LID1=true
    shift
    ;;
  -lid2)
    LID2=true
    shift
    ;;
  -limited)
    LIMITED=true
    shift
    ;;
  -compute-sanitizer-memcheck)
    COMPUTE_SANITIZER=true
    TOOL=memcheck
    shift
    ;;
  -compute-sanitizer-racecheck)
    COMPUTE_SANITIZER=true
    TOOL=racecheck
    shift
    ;;
  -compute-sanitizer-initcheck)
    COMPUTE_SANITIZER=true
    TOOL=initcheck
    shift
    ;;
  -compute-sanitizer-synccheck)
    COMPUTE_SANITIZER=true
    TOOL=synccheck
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

if $LIMITED; then

  export C2H_SEED_COUNT_OVERRIDE=1
  readonly device_mem_GiB=8
  export C2H_DEVICE_MEMORY_LIMIT=$((${device_mem_GiB} * 1024 * 1024 * 1024))
  export C2H_DEBUG_CHECKED_ALLOC_FAILURES=1

  echo "Configuring limited environment:"
  echo "  C2H_SEED_COUNT_OVERRIDE=${C2H_SEED_COUNT_OVERRIDE}"
  echo "  C2H_DEVICE_MEMORY_LIMIT=${C2H_DEVICE_MEMORY_LIMIT} (${device_mem_GiB} GiB)"
  echo "  C2H_DEBUG_CHECKED_ALLOC_FAILURES=${C2H_DEBUG_CHECKED_ALLOC_FAILURES}"
  echo
fi

source "${ci_dir}/build_common.sh"

print_environment_details

./build_cub.sh "$@"

if $NO_LID; then
  PRESETS=("cub-nolid-cpp$CXX_STANDARD")
elif $LID0; then
  PRESETS=("cub-lid0-cpp$CXX_STANDARD")
elif $LID1; then
  PRESETS=("cub-lid1-cpp$CXX_STANDARD")
elif $LID2; then
  PRESETS=("cub-lid2-cpp$CXX_STANDARD")
else
  PRESETS=("cub-cpp$CXX_STANDARD")
fi

if $COMPUTE_SANITIZER; then
  echo "Setting CCCL_TEST_MODE=compute-sanitizer-${TOOL}"
  export CCCL_TEST_MODE=compute-sanitizer-${TOOL}
  echo "Setting C2H_SEED_COUNT_OVERRIDE=1"
  export C2H_SEED_COUNT_OVERRIDE=1
fi

for PRESET in ${PRESETS[@]}; do
  test_preset "CUB (${PRESET})" ${PRESET}
done

print_time_summary
