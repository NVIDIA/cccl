#!/bin/bash

set -euo pipefail

NO_LID=false
LID0=false
LID1=false
LID2=false

ci_dir=$(dirname "$0")

new_args=$("${ci_dir}/util/extract_switches.sh" -no-lid -lid0 -lid1 -lid2 -- "$@")
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

source "${ci_dir}/build_common.sh"

print_environment_details

if true; then # [ -n "${GITHUB_ACTIONS:-}" ]; then
  sudo apt update
  sudo apt install -y pbzip2
  start_time=$SECONDS

  pushd .. > /dev/null
  echo "Unpacking output artifact in $(pwd)"
  ls -lh output_artifact.tar.bz2
  md5sum output_artifact.tar.bz2
  pbzip2 -dc output_artifact.tar.bz2 | tar xvf -
  popd > /dev/null
  echo "Extracting output artifact took $((SECONDS - start_time)) seconds"
else
  ./build_cub.sh "$@"
fi

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

for PRESET in ${PRESETS[@]}; do
  test_preset "CUB (${PRESET})" ${PRESET}
done

print_time_summary
