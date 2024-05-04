#!/bin/bash

set -euo pipefail

NO_LID=
LID0=
LID1=
LID2=

new_args=$(ci/util/extract_switches.sh -no-lid -lid0 -lid1 -lid2 -- "$@")
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

./build_cub.sh "$@"

PRESETS=(
  "cub-nolid-cpp$CXX_STANDARD"
  "cub-lid0-cpp$CXX_STANDARD"
  "cub-lid1-cpp$CXX_STANDARD"
  "cub-lid2-cpp$CXX_STANDARD"
)

if [ -n "$NO_LID" ]; then
  PRESETS=("cub-nolid-cpp$CXX_STANDARD")
elif [ -n "$LID0" ]; then
  PRESETS=("cub-lid0-cpp$CXX_STANDARD")
elif [ -n "$LID1" ]; then
  PRESETS=("cub-lid1-cpp$CXX_STANDARD")
elif [ -n "$LID2" ]; then
  PRESETS=("cub-lid2-cpp$CXX_STANDARD")
fi

for PRESET in ${PRESETS[@]}; do
  test_preset "CUB (${PRESET})" ${PRESET}
done

print_time_summary
